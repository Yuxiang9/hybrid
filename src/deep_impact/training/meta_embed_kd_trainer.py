import torch
import torch.nn.functional as F
from .trainer import Trainer
from .distil_trainer import DistilMarginMSE, DistilKLLoss, DistilTrainer


class MetaEmbedKDTrainer(Trainer):

    def __init__(
        self,
        model,
        optimizer,
        train_data,
        checkpoint_dir,
        batch_size,
        save_every,
        save_best=True,
        seed=42,
        gradient_accumulation_steps=1,
        evaluator=None,
        eval_every=500,
        lambda_di=1.0,
        lambda_li=1.0,
        lambda_kd=1.0,
        margin=0.2,
        temperature=1.0,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self.lambda_di = lambda_di
        self.lambda_li = lambda_li
        self.lambda_kd = lambda_kd
        self.margin = margin
        self.temperature = temperature
        self.reset_tracking()

    def reset_tracking(self):
        self.sparse_loss_sum = 0
        self.dense_loss_sum = 0
        self.kd_loss_sum = 0
        self.sparse_loss_count = 0
        self.dense_loss_count = 0
        self.kd_loss_count = 0

    # ===========================================================
    # Forward passes (batched!)
    # ===========================================================
    def get_output_scores(self, batch):

        device = self.device
        B, D, L = batch["documents"].shape

        # ---- Encode queries ----
        q_sparse, q_dense = self.model.module(
            batch["queries"].to(device),
            batch["query_masks"].to(device),
            torch.zeros_like(batch["queries"]).to(device),
            return_dense_embeddings=True,
        )

        # ---- Encode ALL documents once ----
        flat_docs = batch["documents"].reshape(B * D, L).to(device)
        flat_masks = batch["doc_masks"].reshape(B * D, L).to(device)
        flat_types = torch.zeros_like(flat_docs)

        d_sparse, d_dense = self.model.module(
            flat_docs,
            flat_masks,
            flat_types,
            return_dense_embeddings=True,
        )
        d_sparse = d_sparse.reshape(B, D, L, 1)
        d_dense = d_dense.reshape(B, D, L, -1)

        # ==== masks ====
        q_sparse_mask = batch["query_sparse_masks"].to(device)
        q_dense_mask = batch["query_dense_masks"].squeeze(-1).to(device)
        d_dense_mask = batch["doc_dense_masks"].squeeze(-1).to(device)

        # ===========================================================
        # Sparse score: sum of impact scores for matching tokens
        # ===========================================================
        q_sparse = q_sparse.float()
        d_sparse = d_sparse.float()
        q_sparse_mask = q_sparse_mask.float()

        matched = (flat_docs.unsqueeze(1) == batch["queries"].unsqueeze(2))  # [B, Lq, Ld]
        sparse_score = (matched * d_sparse.squeeze(-1).unsqueeze(1)).sum(dim=[1,2])

        # ===========================================================
        # Dense score: fully batched MaxSim
        # ===========================================================
        dense_scores = torch.zeros(B, D, device=device)
        temp = self.model.module.temperature

        for d_i in range(D):
            sim = torch.matmul(
                q_dense,                                    # (B, Lq, dim)
                d_dense[:, d_i].transpose(1, 2)            # (B, dim, Ld)
            ) / temp
            sim = sim.masked_fill(~d_dense_mask[:, d_i].unsqueeze(1).bool(), -1e9)

            max_sim, _ = sim.max(dim=2)
            max_sim = max_sim.masked_fill(~q_dense_mask.bool(), 0.0)
            dense_scores[:, d_i] = max_sim.sum(dim=1)

        # ===========================================================
        # Combined score (DDP-safe)
        # ===========================================================
        ws = torch.exp(self.model.module.log_sparse_weight)
        wd = torch.exp(self.model.module.log_dense_weight)
        ws = ws / (ws + wd)
        wd = wd / (ws + wd)


        combined_scores = ws * sparse_scores + wd * dense_scores

        # Teacher = combined student (stop gradient)
        teacher_scores = combined_scores.detach()

        return sparse_scores, dense_scores, combined_scores, teacher_scores

    # ===========================================================
    # Loss functions
    # ===========================================================
    def compute_deepimpact_loss(self, scores):
        pos = scores[:, 0]
        neg = scores[:, 1:]
        loss = torch.clamp(self.margin + neg - pos.unsqueeze(1), min=0)
        return loss.mean()

    def compute_infonce_loss(self, scores):
        return F.cross_entropy(
            scores / self.temperature,
            torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        )

    def compute_kd_loss(self, teacher, sparse, dense):
        teacher_p = F.softmax(teacher / self.temperature, dim=1)
        sparse_log = F.log_softmax(sparse / self.temperature, dim=1)
        dense_log  = F.log_softmax(dense  / self.temperature, dim=1)

        kd_sp = -(teacher_p * sparse_log).sum(dim=1).mean()
        kd_dn = -(teacher_p * dense_log ).sum(dim=1).mean()
        return 0.5 * (kd_sp + kd_dn)

    # ===========================================================
    def evaluate_loss(self, outputs, batch):

        sparse, dense, combined, teacher = outputs

        loss_di = self.compute_deepimpact_loss(sparse)
        loss_li = self.compute_infonce_loss(dense)
        loss_kd = self.compute_kd_loss(teacher, sparse, dense)

        loss = (
            self.lambda_di * loss_di +
            self.lambda_li * loss_li +
            self.lambda_kd * loss_kd
        )

        # Track
        self.sparse_loss_sum += loss_di.item()
        self.dense_loss_sum += loss_li.item()
        self.kd_loss_sum += loss_kd.item()
        self.sparse_loss_count += 1
        self.dense_loss_count += 1
        self.kd_loss_count += 1

        return loss

    def train(self):
        self.reset_tracking()
        super().train()

    def _log_batch(self, idx, loss):
        if idx % 10 == 0:
            print(f"[Batch {idx}] Loss={loss:.4f} "
                  f"DI={self.sparse_loss_sum/self.sparse_loss_count:.4f} "
                  f"LI={self.dense_loss_sum/self.dense_loss_count:.4f} "
                  f"KD={self.kd_loss_sum/self.kd_loss_count:.4f}")


def meta_embed_kd_collate_fn(
    batch,
    model_cls,
    max_length: int = 256,
    num_hard_negatives: int = 8,
    top_k_hard: int = 4,
):
    """
    Collate function for MetaEmbed-KD training.
    Adds expansion tokens [exp0 ... expN] to every document BEFORE padding.
    """

    import random
    import torch

    tokenizer = model_cls.tokenizer
    exp_id = model_cls.exp_id
    num_exp = model_cls.num_expansion_slots
    if exp_id is None:
        raise RuntimeError("Expansion tokens not found in vocabulary!")

    queries = []
    all_docs = []
    all_scores = []

    # ===============================
    # 1. NEGATIVE SAMPLING
    # ===============================
    for query, pid_score_list in batch:
        queries.append(query)

        positive_doc, positive_score = pid_score_list[0]
        negatives = pid_score_list[1:]

        if len(negatives) <= num_hard_negatives:
            selected = negatives
        else:
            # top-K hardest
            top_h = negatives[:top_k_hard]
            # random sample from remaining
            remaining = negatives[top_k_hard:]
            n_rand = num_hard_negatives - top_k_hard
            rnd = random.sample(remaining, n_rand)
            selected = top_h + rnd

        docs = [positive_doc] + [doc for doc, _ in selected]
        scores = [positive_score] + [score for _, score in selected]

        all_docs.append(docs)
        all_scores.append(scores)

    B = len(queries)
    D = len(all_docs[0])  # ensure same number

    # ===============================
    # 2. TOKENIZE QUERIES
    # ===============================
    # tokenizer.enable_truncation(max_length=max_length)
    # tokenizer.enable_padding(length=max_length)
    tokenizer.model_max_length = max_length

    enc = tokenizer.batch_encode_plus(
        queries,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    query_ids = enc["input_ids"].clone()
    query_masks = enc["attention_mask"]

    # ===============================
    # 3. TOKENIZE DOCUMENTS + append expansion tokens
    # ===============================
    doc_ids = torch.zeros(B, D, max_length, dtype=torch.long)
    doc_masks = torch.zeros(B, D, max_length, dtype=torch.long)

    for b in range(B):
        for d in range(D):

            doc_text = all_docs[b][d]

            # Encode original doc
            enc = tokenizer(
                doc_text,
                padding=False,
                truncation=False,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors=None,   # return python lists, not tensors
            )
            ids = enc["input_ids"]
            mask = enc["attention_mask"]

            # TRUNCATE BEFORE adding expansion tokens
            max_base_len = max_length - num_exp
            if len(ids) > max_base_len:
                ids = ids[:max_base_len]
                mask = mask[:max_base_len]
            
            ids = ids + [exp_id] * num_exp
            mask = mask + [1] * num_exp

            # ==========================================================
            # 3. PAD TO max_length (never truncate again)
            # ==========================================================
            if len(ids) < max_length:
                pad_len = max_length - len(ids)
                pad_id = tokenizer.pad_token_id
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len

            # Safety check — should never trigger
            if len(ids) != max_length:
                raise RuntimeError(
                    f"Document length mismatch after expansion-token patch: "
                    f"{len(ids)} != {max_length}"
                )

            doc_ids[b, d] = torch.tensor(ids, dtype=torch.long)
            doc_masks[b, d] = torch.tensor(mask, dtype=torch.long)

    # ===============================
    # 4. BUILD MASKS FOR SPARSE & DENSE HEADS
    # ===============================
    # Expansion tokens occupy the LAST num_exp IDs in the vocabulary
    vocab_size = len(tokenizer)

    # new logic: dense mask = equals exp_id
    # use *all query tokens* for dense MaxSim
    query_dense_masks = (query_ids != tokenizer.pad_token_id).float().unsqueeze(-1)
    query_dense_masks = query_dense_masks.unsqueeze(-1).float()
    query_sparse_masks = 1.0 - query_dense_masks
    doc_dense_masks = (doc_ids == exp_id).unsqueeze(-1).float()

    # Scores (teacher from cross-encoder, but we DO NOT use them for KD teacher)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float)

    return {
        "queries": query_ids,
        "documents": doc_ids,
        "query_masks": query_masks,
        "doc_masks": doc_masks,
        "query_sparse_masks": query_sparse_masks,
        "query_dense_masks": query_dense_masks,
        "doc_dense_masks": doc_dense_masks,
        "scores": scores_tensor,
    }

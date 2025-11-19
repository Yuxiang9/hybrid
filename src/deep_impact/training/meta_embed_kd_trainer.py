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
    # Forward passes using encoded_list (consistent with other trainers)
    # ===========================================================
    def get_output_scores(self, batch):
        """
        Compute sparse, dense, and combined scores using encoded_list format.
        
        Args:
            batch: Dictionary with:
                - encoded_list: List of encoded documents with expansion tokens
                - masks: Sparse masks for overlapping terms [B*D, L, 1]
                - doc_dense_masks: Dense masks for expansion tokens [B*D, L, 1]
                - query_dense_masks: Dense masks for query tokens [B, L, 1]
                - num_docs_per_query: Number of documents per query
                
        Returns:
            Tuple of (sparse_scores, dense_scores, combined_scores, teacher_scores)
            Each tensor has shape [B, D] where B=batch_size, D=num_docs_per_query
        """
        device = self.device
        
        # Extract batch info
        D = batch['num_docs_per_query']  # Number of documents per query
        num_total = len(batch['encoded_list'])  # B * D
        B = num_total // D  # Batch size
        
        # ===========================================================
        # 1. ENCODE DOCUMENTS using encoded_list
        # ===========================================================
        # Use base class method to extract tensors from encoded_list
        d_input_ids, d_attention_mask, d_type_ids = self.get_input_tensors(batch['encoded_list'])
        
        # Forward pass through model to get sparse scores and dense embeddings
        d_sparse, d_dense = self.model(
            d_input_ids,
            d_attention_mask,
            d_type_ids,
            return_dense_embeddings=True,
        )
        # d_sparse: [B*D, L, 1]
        # d_dense: [B*D, L, dense_dim]
        
        # Reshape to [B, D, L, ...]
        L = d_sparse.size(1)
        d_sparse = d_sparse.view(B, D, L, 1)
        d_dense = d_dense.view(B, D, L, -1)
        
        # ===========================================================
        # 2. ENCODE QUERIES using query_encoded_list
        # ===========================================================
        # Extract query tensors from query_encoded_list
        q_input_ids, q_attention_mask, q_type_ids = self.get_input_tensors(batch['query_encoded_list'])
        
        # Forward pass through model to get query sparse scores and dense embeddings
        q_sparse, q_dense = self.model(
            q_input_ids,
            q_attention_mask,
            q_type_ids,
            return_dense_embeddings=True,
        )
        # q_sparse: [B, L, 1]
        # q_dense: [B, L, dense_dim]
        
        # Get masks
        sparse_masks = batch['masks'].to(device).view(B, D, L, 1)  # [B, D, L, 1]
        d_dense_mask = batch['doc_dense_masks'].to(device).view(B, D, L, 1).squeeze(-1)  # [B, D, L]
        q_dense_mask = batch['query_dense_masks'].to(device).squeeze(-1)  # [B, L]
        
        # ===========================================================
        # 3. COMPUTE SPARSE SCORES
        # ===========================================================
        # Sparse score: sum of document impact scores for tokens matching query terms
        # sparse_masks[b, d] marks which document tokens match the query
        sparse_scores = (sparse_masks * d_sparse).sum(dim=2).squeeze(-1)  # [B, D]
        
        # ===========================================================
        # 4. COMPUTE DENSE SCORES using MaxSim (late interaction)
        # ===========================================================
        # Dense score: MaxSim between query and document expansion embeddings
        # Use a list to avoid inplace operations that break gradient computation

        temp = self.model.module.temperature
        
        # q_dense:      [B, Lq, dim]
        # d_dense:      [B, D, Ld, dim]
        # q_dense_mask: [B, Lq]
        # d_dense_mask: [B, D, Ld]
        B, D, Ld, dim = d_dense.shape
        Lq = q_dense.shape[1]

        # 1. Expand query embeddings for broadcasting:
        q = q_dense.unsqueeze(1)                  # [B, 1, Lq, dim]
        d = d_dense                               # [B, D, Ld, dim]

        # 2. Compute batched similarity
        sim = torch.matmul(q, d.transpose(2, 3)) / temp  
        # sim: [B, D, Lq, Ld]


        # 3. Apply masks
        sim = sim.masked_fill(~d_dense_mask.unsqueeze(2).bool(), -1e4)
        sim = sim.masked_fill(~q_dense_mask.unsqueeze(1).unsqueeze(-1).bool(), -1e4)

        # 4. MaxSim over document tokens
        max_sim, _ = sim.max(dim=3)               # [B, D, Lq]

        # 5. Zero out padding query tokens
        max_sim = max_sim.masked_fill(~q_dense_mask.unsqueeze(1).bool(), 0.0)

        # 6. Sum over query dimension
        dense_scores = max_sim.sum(dim=2)         # [B, D]      
          
        # ===========================================================
        # 5. COMBINED SCORES
        # ===========================================================
        ws = torch.exp(self.model.module.log_sparse_weight)
        wd = torch.exp(self.model.module.log_dense_weight)
        ws = ws / (ws + wd)
        wd = wd / (ws + wd)
        
        combined_scores = ws * sparse_scores + wd * dense_scores
        teacher_scores = batch["scores"].view(B, D).to(device)
        
        return sparse_scores, dense_scores, combined_scores, teacher_scores

    # ===========================================================
    # Loss functions (all using KL divergence for distillation)
    # ===========================================================
    def compute_sparse_kd_loss(self, teacher_scores, sparse_scores):
        """
        KL divergence loss for sparse scores only.
        
        Distills teacher knowledge into the sparse (lexical) component.
        Does NOT require positive/negative distinction.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            sparse_scores: Student sparse scores [B, D]
            
        Returns:
            KL divergence loss for sparse component
        """
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        sparse_log = F.log_softmax(sparse_scores / self.temperature, dim=1)
        
        # KL(teacher || sparse)
        loss = -(teacher_p * sparse_log).sum(dim=1).mean()
        return loss

    def compute_dense_kd_loss(self, teacher_scores, dense_scores):
        """
        KL divergence loss for dense scores only.
        
        Distills teacher knowledge into the dense (semantic) component.
        Does NOT require positive/negative distinction.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            dense_scores: Student dense scores [B, D]
            
        Returns:
            KL divergence loss for dense component
        """
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        dense_log = F.log_softmax(dense_scores / self.temperature, dim=1)
        
        # KL(teacher || dense)
        loss = -(teacher_p * dense_log).sum(dim=1).mean()
        return loss

    def compute_combined_kd_loss(self, teacher_scores, sparse_scores, dense_scores):
        """
        KL divergence loss for the combined (sparse + dense) student scores.
        
        This distills teacher knowledge into the FINAL combined output of the model.
        The student's combined score (sparse + dense) is matched to the teacher.
        
        This is cleaner than separately distilling sparse and dense components,
        as it directly trains the final retrieval score.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            sparse_scores: Student sparse scores [B, D]
            dense_scores: Student dense scores [B, D]
            
        Returns:
            KL divergence loss for combined student scores
        """
        # Combine sparse and dense scores (simple addition)
        combined_student_scores = sparse_scores + dense_scores
        
        # Compute KL divergence between teacher and combined student
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        combined_log = F.log_softmax(combined_student_scores / self.temperature, dim=1)
        
        # KL(teacher || combined_student)
        loss = -(teacher_p * combined_log).sum(dim=1).mean()
        
        return loss

    # ===========================================================
    def evaluate_loss(self, outputs, batch):
        """
        Compute combined loss with configurable weights.
        
        All losses use KL divergence for knowledge distillation:
        - lambda_di: Weight for sparse component KD loss (teacher → sparse only)
        - lambda_li: Weight for dense component KD loss (teacher → dense only)
        - lambda_kd: Weight for combined KD loss (teacher → sparse+dense combined)
        
        RECOMMENDED: Use ONLY lambda_kd=1.0 for most cases.
        This distills the teacher into the FINAL combined output (sparse+dense),
        which is what you actually use for retrieval.
        
        Alternative usage (if you want to emphasize individual components):
        - lambda_di=0.3, lambda_li=0.3, lambda_kd=0.4
        
        Note: lambda_kd uses (sparse + dense) as the student score, which is
        different from (lambda_di + lambda_li) which treats them separately.
        """
        sparse, dense, combined, teacher = outputs
        
        # Use teacher scores from batch (cross-encoder scores)
        teacher_scores = batch['scores'].view(sparse.shape).to(self.device)

        # Compute losses only if their weights are non-zero
        if self.lambda_di > 0:
            loss_di = self.compute_sparse_kd_loss(teacher_scores, sparse)
        else:
            loss_di = torch.tensor(0.0, device=self.device)
            
        if self.lambda_li > 0:
            loss_li = self.compute_dense_kd_loss(teacher_scores, dense)
        else:
            loss_li = torch.tensor(0.0, device=self.device)
            
        if self.lambda_kd > 0:
            loss_kd = self.compute_combined_kd_loss(teacher_scores, sparse, dense)
        else:
            loss_kd = torch.tensor(0.0, device=self.device)

        loss = (
            self.lambda_di * loss_di +
            self.lambda_li * loss_li +
            self.lambda_kd * loss_kd
        )

        # Track (only if non-zero)
        if self.lambda_di > 0:
            self.sparse_loss_sum += loss_di.item()
            self.sparse_loss_count += 1
        if self.lambda_li > 0:
            self.dense_loss_sum += loss_li.item()
            self.dense_loss_count += 1
        if self.lambda_kd > 0:
            self.kd_loss_sum += loss_kd.item()
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
# def meta_embed_kd_collate_fn(
#     batch,
#     model_cls,
#     max_length: int = 256,
#     num_hard_negatives: int = 8,
#     top_k_hard: int = 4,
#     qrels_path: Optional[Union[str, Path]] = None,
# ):
#     """
#     Collate function for MetaEmbed-KD training.
#     Adds expansion tokens [exp0 ... expN] to every document BEFORE padding.
#     """

#     import random
#     import torch

#     tokenizer = model_cls.tokenizer
#     exp_id = model_cls.exp_id
#     num_exp = model_cls.num_expansion_slots
#     if exp_id is None:
#         raise RuntimeError("Expansion tokens not found in vocabulary!")

#     queries = []
#     all_docs = []
#     all_scores = []

#     # ===============================
#     # 1. NEGATIVE SAMPLING
#     # ===============================
#     for query, pid_score_list in batch:
#         queries.append(query)

#         positive_doc, positive_score = pid_score_list[0]
#         negatives = pid_score_list[1:]

#         if len(negatives) <= num_hard_negatives:
#             selected = negatives
#         else:
#             # top-K hardest
#             top_h = negatives[:top_k_hard]
#             # random sample from remaining
#             remaining = negatives[top_k_hard:]
#             n_rand = num_hard_negatives - top_k_hard
#             rnd = random.sample(remaining, n_rand)
#             selected = top_h + rnd

#         docs = [positive_doc] + [doc for doc, _ in selected]
#         scores = [positive_score] + [score for _, score in selected]

#         all_docs.append(docs)
#         all_scores.append(scores)

#     B = len(queries)
#     D = len(all_docs[0])  # ensure same number

#     # ===============================
#     # 2. TOKENIZE QUERIES
#     # ===============================
#     # tokenizer.enable_truncation(max_length=max_length)
#     # tokenizer.enable_padding(length=max_length)
#     tokenizer.model_max_length = max_length

#     enc = tokenizer.batch_encode_plus(
#         queries,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt",
#     )
#     query_ids = enc["input_ids"].clone()
#     query_masks = enc["attention_mask"]

#     # ===============================
#     # 3. TOKENIZE DOCUMENTS + append expansion tokens
#     # ===============================
#     doc_ids = torch.zeros(B, D, max_length, dtype=torch.long)
#     doc_masks = torch.zeros(B, D, max_length, dtype=torch.long)

#     for b in range(B):
#         for d in range(D):

#             doc_text = all_docs[b][d]

#             # Encode original doc
#             enc = tokenizer(
#                 doc_text,
#                 padding=False,
#                 truncation=False,
#                 max_length=max_length,
#                 return_attention_mask=True,
#                 return_tensors=None,   # return python lists, not tensors
#             )
#             ids = enc["input_ids"]
#             mask = enc["attention_mask"]

#             # TRUNCATE BEFORE adding expansion tokens
#             max_base_len = max_length - num_exp
#             if len(ids) > max_base_len:
#                 ids = ids[:max_base_len]
#                 mask = mask[:max_base_len]
            
#             ids = ids + [exp_id] * num_exp
#             mask = mask + [1] * num_exp

#             # ==========================================================
#             # 3. PAD TO max_length (never truncate again)
#             # ==========================================================
#             if len(ids) < max_length:
#                 pad_len = max_length - len(ids)
#                 pad_id = tokenizer.pad_token_id
#                 ids = ids + [pad_id] * pad_len
#                 mask = mask + [0] * pad_len

#             # Safety check — should never trigger
#             if len(ids) != max_length:
#                 raise RuntimeError(
#                     f"Document length mismatch after expansion-token patch: "
#                     f"{len(ids)} != {max_length}"
#                 )

#             doc_ids[b, d] = torch.tensor(ids, dtype=torch.long)
#             doc_masks[b, d] = torch.tensor(mask, dtype=torch.long)

#     # ===============================
#     # 4. BUILD MASKS FOR SPARSE & DENSE HEADS
#     # ===============================
#     # Dense masks: For late interaction MaxSim
#     # - Query: use all non-padding tokens
#     # - Document: use only expansion tokens
#     query_dense_masks = (query_ids != tokenizer.pad_token_id).float().unsqueeze(-1)  # [B, L, 1]
#     doc_dense_masks = (doc_ids == exp_id).unsqueeze(-1).float()  # [B, D, L, 1]
    
#     # Sparse masks: For lexical matching (overlap checking)
#     # Note: This is a placeholder - actual overlap is computed dynamically in get_output_scores
#     # We just need to mark all non-padding tokens here
#     query_sparse_masks = (query_ids != tokenizer.pad_token_id).float().unsqueeze(-1)  # [B, L, 1]

#     # Scores (teacher from cross-encoder, but we DO NOT use them for KD teacher)
#     scores_tensor = torch.tensor(all_scores, dtype=torch.float)

#     return {
#         "queries": query_ids,
#         "documents": doc_ids,
#         "query_masks": query_masks,
#         "doc_masks": doc_masks,
#         "query_sparse_masks": query_sparse_masks,
#         "query_dense_masks": query_dense_masks,
#         "doc_dense_masks": doc_dense_masks,
#         "scores": scores_tensor,
#     }



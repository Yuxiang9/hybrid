"""
MetaEmbed KD Trainer: Joint Training with Knowledge Distillation

Implements the loss function from DeeperImpact paper (Section 2.2.3):
L = λ_DI * L_DeepImpact + λ_LI * L_late-int + λ_KD * L_KD

Where:
1. L_DeepImpact: Margin loss on sparse scores (max(0, m + s(q,d-) - s(q,d+)))
2. L_late-int: InfoNCE contrastive loss on dense scores  
3. L_KD: Knowledge distillation from combined scores to individual heads

Uses the existing DistillationScores dataset (from DeeperImpact) which provides:
- Teacher scores from ms-marco-MiniLM-L-6-v2 cross-encoder
- 1 positive + ~50 hard negatives per query
- Teacher scores for each document
"""

import torch
import torch.nn.functional as F
from typing import Union, Dict, Tuple
from pathlib import Path

from .trainer import Trainer
from ..models.meta_embed_impact import MetaEmbedDeepImpact


class MetaEmbedKDTrainer(Trainer):
    """
    Trainer for MetaEmbed with knowledge distillation.
    
    Combines three loss components:
    1. DeepImpact margin loss for sparse head
    2. InfoNCE contrastive loss for dense head
    3. Knowledge distillation from combined to individual heads
    """
    
    def __init__(
        self,
        model,
        optimizer,
        train_data,
        checkpoint_dir: Union[str, Path],
        batch_size: int,
        save_every: int,
        save_best: bool = True,
        seed: int = 42,
        gradient_accumulation_steps: int = 1,
        evaluator=None,
        eval_every: int = 500,
        # Loss weights
        lambda_di: float = 1.0,
        lambda_li: float = 1.0,
        lambda_kd: float = 1.0,
        # Margin for DeepImpact loss
        margin: float = 0.2,
        # Temperature for InfoNCE and KD
        temperature: float = 1.0,
    ):
        """
        Initialize trainer with loss weights and hyperparameters.
        
        Args:
            model: MetaEmbedDeepImpact model
            optimizer: Optimizer
            train_data: DataLoader
            checkpoint_dir: Directory to save checkpoints
            batch_size: Batch size per GPU
            save_every: Save checkpoint every n steps
            save_best: Whether to save best model
            seed: Random seed
            gradient_accumulation_steps: Steps for gradient accumulation
            evaluator: Evaluator object
            eval_every: Evaluate every n steps
            lambda_di: Weight for DeepImpact margin loss
            lambda_li: Weight for late-interaction InfoNCE loss
            lambda_kd: Weight for knowledge distillation loss
            margin: Margin for DeepImpact loss (default: 0.2)
            temperature: Temperature for InfoNCE and KD (default: 1.0)
        """
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
        
        # Statistics tracking
        self.sparse_loss_sum = 0.0
        self.dense_loss_sum = 0.0
        self.kd_loss_sum = 0.0
        self.sparse_loss_count = 0
        self.dense_loss_count = 0
        self.kd_loss_count = 0
    
    def get_output_scores(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sparse, dense, and combined scores for all query-document pairs.
        
        Args:
            batch: Dictionary containing:
                - queries: [batch_size, seq_len]
                - documents: [batch_size, num_docs, seq_len]
                - query_masks: [batch_size, seq_len]
                - doc_masks: [batch_size, num_docs, seq_len]
                - query_sparse_masks: [batch_size, seq_len, 1]
                - query_dense_masks: [batch_size, seq_len, 1]
                - doc_dense_masks: [batch_size, num_docs, seq_len, 1]
                - scores: [batch_size, num_docs] - teacher scores
        
        Returns:
            Tuple of (sparse_scores, dense_scores, combined_scores, teacher_scores)
            Each tensor has shape [batch_size, num_docs]
        """
        batch_size = batch['queries'].shape[0]
        num_docs = batch['documents'].shape[1]
        
        # Encode queries (once per query)
        q_input_ids = batch['queries'].to(self.gpu_id)
        q_attention_mask = batch['query_masks'].to(self.gpu_id)
        q_token_type_ids = torch.zeros_like(q_input_ids)  # Required by BERT, all zeros for single segment
        
        q_sparse_scores, q_dense_embeddings = self.model.module(
            q_input_ids, q_attention_mask, q_token_type_ids, return_dense_embeddings=True
        )
        
        # Get masks
        query_sparse_masks = batch['query_sparse_masks'].to(self.gpu_id)  # [batch_size, seq_len, 1]
        query_dense_masks = batch['query_dense_masks'].to(self.gpu_id).squeeze(-1)  # [batch_size, seq_len]
        doc_dense_masks = batch['doc_dense_masks'].to(self.gpu_id).squeeze(-1)  # [batch_size, num_docs, seq_len]
        
        # CRITICAL: Extract parameters once to avoid DDP issues
        weights = F.softmax(torch.stack([self.model.module.log_sparse_weight, 
                                          self.model.module.log_dense_weight]), dim=0)
        w_sparse, w_dense = weights[0], weights[1]
        temperature = self.model.module.temperature
        
        # Initialize score tensors
        sparse_scores = torch.zeros(batch_size, num_docs, device=self.gpu_id)
        dense_scores = torch.zeros(batch_size, num_docs, device=self.gpu_id)
        
        # Process each query-document pair
        for b in range(batch_size):
            q_sparse = q_sparse_scores[b:b+1]  # [1, seq_len, 1]
            q_dense = q_dense_embeddings[b:b+1]  # [1, seq_len, dense_dim]
            q_sparse_mask = query_sparse_masks[b:b+1]  # [1, seq_len, 1]
            q_dense_mask = query_dense_masks[b:b+1]  # [1, seq_len]
            
            for d in range(num_docs):
                # Encode document
                d_input_ids = batch['documents'][b, d].unsqueeze(0).to(self.gpu_id)
                d_attention_mask = batch['doc_masks'][b, d].unsqueeze(0).to(self.gpu_id)
                d_token_type_ids = torch.zeros_like(d_input_ids)  # Required by BERT, all zeros for single segment
                
                d_sparse_scores, d_dense_embeddings = self.model.module(
                    d_input_ids, d_attention_mask, d_token_type_ids, return_dense_embeddings=True
                )
                
                d_dense_mask = doc_dense_masks[b, d].unsqueeze(0)  # [1, seq_len]
                
                # Compute sparse score (sum of masked impact scores)
                sparse_score_raw = (q_sparse_mask * d_sparse_scores).sum(dim=1).squeeze(-1)
                
                # Compute dense score (MaxSim late interaction)
                dense_score_raw = self.model.module.compute_late_interaction_score(
                    q_dense, d_dense_embeddings, q_dense_mask, d_dense_mask, temperature
                )
                
                sparse_scores[b, d] = sparse_score_raw[0]
                dense_scores[b, d] = dense_score_raw[0]
        
        # Apply combination weights
        combined_scores = w_sparse * sparse_scores + w_dense * dense_scores
        
        # Get teacher scores
        teacher_scores = batch['scores'].to(self.gpu_id)
        
        return sparse_scores, dense_scores, combined_scores, teacher_scores
    
    def compute_deepimpact_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute DeepImpact margin loss: max(0, m + s(q,d-) - s(q,d+))
        
        Args:
            scores: Tensor of shape [batch_size, num_docs] where scores[:, 0] is positive
        
        Returns:
            Scalar loss
        """
        batch_size = scores.shape[0]
        num_negatives = scores.shape[1] - 1
        
        positive_scores = scores[:, 0]  # [batch_size]
        negative_scores = scores[:, 1:]  # [batch_size, num_negatives]
        
        # Compute margin loss for each negative
        # max(0, m + s(q,d-) - s(q,d+))
        losses = torch.clamp(self.margin + negative_scores - positive_scores.unsqueeze(1), min=0.0)
        
        # Average over all negatives and batch
        loss = losses.mean()
        
        return loss
    
    def compute_infonce_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss: -log(exp(s+) / sum(exp(s_j)))
        
        Args:
            scores: Tensor of shape [batch_size, num_docs] where scores[:, 0] is positive
        
        Returns:
            Scalar loss
        """
        # scores: [batch_size, num_docs]
        # Scale by temperature
        scores = scores / self.temperature
        
        # InfoNCE: -log(exp(s_pos) / sum(exp(s_all)))
        # This is equivalent to cross-entropy where target is index 0
        loss = F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device))
        
        return loss
    
    def compute_kd_loss(
        self, 
        teacher_scores: torch.Tensor,
        student_sparse_scores: torch.Tensor,
        student_dense_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        The combined score (teacher) teaches both individual heads (students):
        L_KD = 0.5 * [CE(softmax(teacher), softmax(sparse)) + CE(softmax(teacher), softmax(dense))]
        
        Args:
            teacher_scores: Combined scores [batch_size, num_docs]
            student_sparse_scores: Sparse scores [batch_size, num_docs]
            student_dense_scores: Dense scores [batch_size, num_docs]
        
        Returns:
            Scalar loss
        """
        # Convert scores to log probabilities via softmax
        teacher_logprobs = F.log_softmax(teacher_scores / self.temperature, dim=1)
        sparse_logprobs = F.log_softmax(student_sparse_scores / self.temperature, dim=1)
        dense_logprobs = F.log_softmax(student_dense_scores / self.temperature, dim=1)
        
        # Compute KL divergence: KL(teacher || student)
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        # For cross-entropy: CE(P,Q) = -sum(P * log(Q))
        # We use teacher as target distribution
        
        # Convert teacher to probabilities for target
        teacher_probs = F.softmax(teacher_scores / self.temperature, dim=1)
        
        # Cross-entropy: -sum(teacher_probs * log(student_probs))
        kd_sparse = -(teacher_probs * sparse_logprobs).sum(dim=1).mean()
        kd_dense = -(teacher_probs * dense_logprobs).sum(dim=1).mean()
        
        # Average the two KD losses
        kd_loss = 0.5 * (kd_sparse + kd_dense)
        
        return kd_loss
    
    def evaluate_loss(self, outputs, batch):
        """
        Compute combined loss: L = λ_DI * L_DI + λ_LI * L_LI + λ_KD * L_KD
        
        Args:
            outputs: Tuple of (sparse_scores, dense_scores, combined_scores, teacher_scores)
            batch: Batch dictionary
        
        Returns:
            Combined loss scalar
        """
        sparse_scores, dense_scores, combined_scores, teacher_scores = outputs
        
        # 1. DeepImpact margin loss on sparse scores
        loss_di = self.compute_deepimpact_loss(sparse_scores)
        
        # 2. InfoNCE contrastive loss on dense scores
        loss_li = self.compute_infonce_loss(dense_scores)
        
        # 3. Knowledge distillation from combined to individual heads
        loss_kd = self.compute_kd_loss(combined_scores, sparse_scores, dense_scores)
        
        # Combined loss
        loss = self.lambda_di * loss_di + self.lambda_li * loss_li + self.lambda_kd * loss_kd
        
        # Track statistics
        self.sparse_loss_sum += loss_di.item()
        self.sparse_loss_count += 1
        self.dense_loss_sum += loss_li.item()
        self.dense_loss_count += 1
        self.kd_loss_sum += loss_kd.item()
        self.kd_loss_count += 1
        
        return loss
    
    def train(self):
        """Override train to reset statistics at the beginning."""
        # Reset statistics
        self.sparse_loss_sum = 0.0
        self.dense_loss_sum = 0.0
        self.kd_loss_sum = 0.0
        self.sparse_loss_count = 0
        self.dense_loss_count = 0
        self.kd_loss_count = 0
        
        # Call parent train
        super().train()
    
    def _log_batch(self, batch_idx, loss):
        """Override to log component losses."""
        # Compute averages
        avg_sparse = self.sparse_loss_sum / max(self.sparse_loss_count, 1)
        avg_dense = self.dense_loss_sum / max(self.dense_loss_count, 1)
        avg_kd = self.kd_loss_sum / max(self.kd_loss_count, 1)
        
        # Log with component breakdown
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss={loss:.4f} | "
                  f"L_DI={avg_sparse:.4f} | L_LI={avg_dense:.4f} | L_KD={avg_kd:.4f}")


def meta_embed_kd_collate_fn(
    batch,
    model_cls: MetaEmbedDeepImpact,
    max_length: int = 256,
    num_hard_negatives: int = 8,
    top_k_hard: int = 4,
):
    """
    Collate function for MetaEmbed KD training with distillation scores.
    
    Expects batch items from DistillationScores dataset:
    (query, [(doc1, score1), (doc2, score2), ..., (docN, scoreN)])
    
    The first document has the highest teacher score (positive), rest are negatives.
    This function samples a subset of negatives to reduce memory:
    - Top K hard negatives (highest scores after positive)
    - Random sample from remaining negatives
    
    Args:
        batch: List of (query, [(doc, score), ...]) tuples from DistillationScores
        model_cls: MetaEmbedDeepImpact class (for tokenizer)
        max_length: Maximum sequence length
        num_hard_negatives: Total number of negatives to use (default: 8)
        top_k_hard: Number of top hard negatives to always include (default: 4)
    
    Returns:
        Dictionary with:
            - queries: [batch_size, seq_len]
            - documents: [batch_size, num_docs, seq_len]  (num_docs = 1 + num_hard_negatives)
            - query_masks: [batch_size, seq_len]
            - doc_masks: [batch_size, num_docs, seq_len]
            - query_sparse_masks: [batch_size, seq_len, 1]
            - query_dense_masks: [batch_size, seq_len, 1]
            - doc_dense_masks: [batch_size, num_docs, seq_len, 1]
            - scores: [batch_size, num_docs] - teacher scores
    """
    import random
    
    queries = []
    all_docs = []
    all_scores = []
    
    for idx, (query, doc_score_list) in enumerate(batch):
        queries.append(query)
        
        # doc_score_list is sorted by score (highest first)
        # Index 0: positive document (highest score)
        # Index 1+: negative documents (sorted by decreasing score)
        
        positive_doc, positive_score = doc_score_list[0]
        negatives = doc_score_list[1:]  # All negatives
        
        # Sample negatives intelligently
        if len(negatives) <= num_hard_negatives:
            # If we have fewer negatives than requested, use all
            selected_negatives = negatives
        else:
            # Top K hard negatives (hardest to distinguish from positive)
            top_hard = negatives[:top_k_hard]
            
            # Random sample from remaining negatives
            remaining = negatives[top_k_hard:]
            num_random = min(num_hard_negatives - top_k_hard, len(remaining))
            random_sample = random.sample(remaining, num_random)
            
            # Combine: positive + top_hard + random_sample
            selected_negatives = top_hard + random_sample
        
        # Combine positive and selected negatives
        selected_docs = [positive_doc] + [doc for doc, score in selected_negatives]
        selected_scores = [positive_score] + [score for doc, score in selected_negatives]
        
        all_docs.append(selected_docs)
        all_scores.append(selected_scores)
    
    batch_size = len(queries)
    
    # Ensure all queries have the same number of documents
    # Use the minimum to avoid issues with batching
    num_docs_per_query = [len(docs) for docs in all_docs]
    
    if len(set(num_docs_per_query)) > 1:
        # Different queries have different numbers of docs - use minimum
        num_docs = min(num_docs_per_query)
        # Truncate all to the same length
        all_docs = [docs[:num_docs] for docs in all_docs]
        all_scores = [scores[:num_docs] for scores in all_scores]
    else:
        num_docs = num_docs_per_query[0]
    
    # Ensure tokenizer is configured with correct max_length and padding
    model_cls.tokenizer.enable_truncation(max_length=max_length)
    model_cls.tokenizer.enable_padding(length=max_length)
    
    # Tokenize queries
    try:
        query_encodings = model_cls.tokenizer.encode_batch(queries)
    except Exception as e:
        raise
    
    query_ids = torch.tensor([enc.ids for enc in query_encodings])
    query_masks = torch.tensor([enc.attention_mask for enc in query_encodings])
    
    # Tokenize documents
    doc_ids = torch.zeros(batch_size, num_docs, max_length, dtype=torch.long)
    doc_masks = torch.zeros(batch_size, num_docs, max_length, dtype=torch.long)
    
    for b in range(batch_size):
        for d, doc in enumerate(all_docs[b]):
            enc = model_cls.tokenizer.encode(doc)
            doc_ids[b, d] = torch.tensor(enc.ids)
            doc_masks[b, d] = torch.tensor(enc.attention_mask)
    
    # Create masks for sparse and dense tokens
    # Use len(expansion_tokens) since num_expansion_tokens is instance attribute, not class attribute
    num_exp_tokens = len(model_cls.expansion_tokens)
    expansion_token_start = model_cls.tokenizer.get_vocab_size() - num_exp_tokens
    
    # Query sparse mask: 1 for non-expansion tokens
    query_sparse_masks = (query_ids < expansion_token_start).unsqueeze(-1).float()
    
    # Query dense mask: 1 for expansion tokens  
    query_dense_masks = (query_ids >= expansion_token_start).unsqueeze(-1).float()
    
    # Document dense mask: 1 for expansion tokens
    doc_dense_masks = (doc_ids >= expansion_token_start).unsqueeze(-1).float()
    
    # Teacher scores
    scores_tensor = torch.tensor(all_scores, dtype=torch.float)
    
    return {
        'queries': query_ids,
        'documents': doc_ids,
        'query_masks': query_masks,
        'doc_masks': doc_masks,
        'query_sparse_masks': query_sparse_masks,
        'query_dense_masks': query_dense_masks,
        'doc_dense_masks': doc_dense_masks,
        'scores': scores_tensor,
    }



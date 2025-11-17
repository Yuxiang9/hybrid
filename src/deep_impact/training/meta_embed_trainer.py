"""
MetaEmbed Trainer: Joint Training of Sparse and Dense Components

Implements contrastive learning (InfoNCE) for the hybrid sparse-dense model.
The loss function combines:
1. Sparse component loss (standard DeepImpact contrastive)
2. Dense component loss (late interaction contrastive)
3. Combined score loss (optional)

Following the MetaEmbed paper's approach with hierarchical multi-vector embeddings.
"""

import torch
import torch.nn.functional as F
from typing import Union
from pathlib import Path
import json

from .trainer import Trainer
from ..models.meta_embed_impact import MetaEmbedDeepImpact


class InfoNCELoss:
    """
    InfoNCE loss for contrastive learning.
    
    Implements the loss from Equation (6) in the MetaEmbed paper:
    L = -1/B * Σ log( exp(S(q,d+)) / (exp(S(q,d+)) + Σ exp(S(q,d-)) + exp(S(q,d_hard-))) )
    
    where:
    - q: query
    - d+: positive document
    - d-: in-batch negative documents
    - d_hard-: explicit hard negative
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax (default: 1.0)
        """
        self.temperature = temperature
    
    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            scores: Scores tensor [batch_size, num_candidates] where
                   scores[:, 0] are positive scores
                   scores[:, 1:] are negative scores
                   
        Returns:
            Scalar loss value
        """
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Compute log softmax
        log_probs = F.log_softmax(scores, dim=1)
        
        # Loss is negative log probability of positive (index 0)
        loss = -log_probs[:, 0].mean()
        
        return loss


class CombinedInfoNCELoss:
    """
    Combined InfoNCE loss for sparse, dense, and integrated scores.
    
    Implements the final loss from Equation (7) in MetaEmbed paper:
    L_final = Σ w_g * L_NCE^(g)
    
    In our case, the groups are:
    - g1: sparse scores (lexical matching)
    - g2: dense scores (semantic late interaction)
    - g3: combined scores (hybrid)
    """
    
    def __init__(self, 
                 sparse_weight: float = 1.0,
                 dense_weight: float = 1.0, 
                 combined_weight: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize combined InfoNCE loss.
        
        Args:
            sparse_weight: Weight for sparse component loss (w_sparse)
            dense_weight: Weight for dense component loss (w_dense)
            combined_weight: Weight for combined score loss (w_combined)
            temperature: Temperature for softmax
        """
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.combined_weight = combined_weight
        self.loss_fn = InfoNCELoss(temperature=temperature)
    
    def __call__(self, 
                 sparse_scores: torch.Tensor,
                 dense_scores: torch.Tensor,
                 combined_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss across all score components.
        
        Args:
            sparse_scores: Sparse scores [batch_size, num_candidates]
            dense_scores: Dense scores [batch_size, num_candidates]
            combined_scores: Combined scores [batch_size, num_candidates]
            
        Returns:
            Weighted combined loss
        """
        loss_sparse = self.loss_fn(sparse_scores)
        loss_dense = self.loss_fn(dense_scores)
        loss_combined = self.loss_fn(combined_scores)
        
        total_loss = (
            self.sparse_weight * loss_sparse +
            self.dense_weight * loss_dense +
            self.combined_weight * loss_combined
        )
        
        return total_loss, loss_sparse, loss_dense, loss_combined


class MetaEmbedTrainer(Trainer):
    """
    Trainer for MetaEmbed Deep Impact with joint sparse-dense learning.
    
    This trainer:
    1. Computes sparse scores using impact scores for matching terms
    2. Computes dense scores using MaxSim late interaction on expansion tokens
    3. Computes combined scores as weighted sum
    4. Applies InfoNCE contrastive loss to all three scores
    
    Training uses in-batch negatives:
    - Each query has 1 positive document and 1 hard negative
    - All other positives in the batch serve as additional negatives
    """
    
    def __init__(
        self,
        model: MetaEmbedDeepImpact,
        optimizer: torch.optim.Optimizer,
        train_data,
        checkpoint_dir: Union[str, Path],
        batch_size: int,
        save_every: int,
        save_best: bool = True,
        seed: int = 42,
        gradient_accumulation_steps: int = 1,
        eval_every: int = 500,
        evaluator=None,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        combined_weight: float = 1.0,
        temperature: float = 1.0,
        log_components: bool = True,
    ):
        """
        Initialize MetaEmbed trainer.
        
        Args:
            model: MetaEmbedDeepImpact model
            optimizer: Optimizer
            train_data: DataLoader for training
            checkpoint_dir: Directory for checkpoints
            batch_size: Batch size
            save_every: Save checkpoint every N steps
            save_best: Whether to save best model
            seed: Random seed
            gradient_accumulation_steps: Gradient accumulation steps
            eval_every: Evaluate every N steps
            evaluator: Evaluator for validation
            sparse_weight: Weight for sparse loss component
            dense_weight: Weight for dense loss component
            combined_weight: Weight for combined loss component
            temperature: Temperature for InfoNCE loss
            log_components: Whether to log individual loss components
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
            eval_every=eval_every,
            evaluator=evaluator,
        )
        
        # Override criterion with combined InfoNCE loss
        self.criterion = CombinedInfoNCELoss(
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            combined_weight=combined_weight,
            temperature=temperature,
        )
        
        self.log_components = log_components
        
        # Statistics for monitoring training
        self.loss_stats = {
            'total': 0.0,
            'sparse': 0.0,
            'dense': 0.0,
            'combined': 0.0,
            'count': 0,
        }
    
    def get_output_scores(self, batch):
        """
        Compute sparse, dense, and combined scores for query-document pairs.
        
        This method:
        1. Encodes queries to get sparse scores and dense embeddings
        2. Encodes documents (positive + negatives) to get sparse scores and dense embeddings
        3. Computes sparse, dense, and combined scores using the model
        
        Args:
            batch: Dictionary containing:
                - query_encoded_list: List of encoded queries
                - encoded_list: List of encoded documents (positive, negative for each query)
                - query_sparse_masks: Masks for query terms [batch_size, seq_len, 1]
                - query_dense_masks: Masks for query expansion tokens [batch_size, seq_len]
                - doc_dense_masks: Masks for document expansion tokens [batch_size*2, seq_len]
                
        Returns:
            Tuple of (sparse_scores, dense_scores, combined_scores) each [batch_size, 2]
        """
        batch_size = self.batch_size
        
        # Encode queries
        q_input_ids, q_attention_mask, q_type_ids = self.get_input_tensors(batch['query_encoded_list'])
        q_sparse_scores, q_dense_embeddings = self.model(
            q_input_ids,
            q_attention_mask,
            q_type_ids,
            return_dense_embeddings=True
        )
        
        # Encode documents (positive + negative for each query)
        d_input_ids, d_attention_mask, d_type_ids = self.get_input_tensors(batch['encoded_list'])
        d_sparse_scores, d_dense_embeddings = self.model(
            d_input_ids,
            d_attention_mask,
            d_type_ids,
            return_dense_embeddings=True
        )
        
        # Get masks
        query_sparse_masks = batch['query_sparse_masks'].to(self.gpu_id)  # [batch_size, seq_len, 1]
        query_dense_masks = batch['query_dense_masks'].to(self.gpu_id).squeeze(-1)  # [batch_size, seq_len]
        doc_dense_masks = batch['doc_dense_masks'].to(self.gpu_id).squeeze(-1)  # [batch_size*2, seq_len]
        
        # IMPORTANT: Compute ALL parameters once before the loop to avoid DDP issues
        # If we access self.w1, self.w2, self.temperature multiple times in the loop,
        # DDP thinks they're used multiple times in the computation graph,
        # causing "marked as ready twice" error
        import torch.nn.functional as F
        weights = F.softmax(torch.stack([self.model.module.log_sparse_weight, 
                                          self.model.module.log_dense_weight]), dim=0)
        w_sparse, w_dense = weights[0], weights[1]
        # ALSO extract temperature (used in compute_late_interaction_score)
        temperature = self.model.module.temperature
        
        # Compute scores for each query-document pair
        sparse_scores_list = []
        dense_scores_list = []
        combined_scores_list = []
        
        for i in range(batch_size):
            # Get query representations (same for positive and negative)
            q_sparse = q_sparse_scores[i:i+1]  # [1, seq_len, 1]
            q_dense = q_dense_embeddings[i:i+1]  # [1, seq_len, dense_dim]
            q_sparse_mask = query_sparse_masks[i:i+1]  # [1, seq_len, 1]
            q_dense_mask = query_dense_masks[i:i+1]  # [1, seq_len]
            
            # Get positive and negative document representations
            d_pos_idx = i * 2
            d_neg_idx = i * 2 + 1
            
            d_sparse_pos = d_sparse_scores[d_pos_idx:d_pos_idx+1]  # [1, seq_len, 1]
            d_sparse_neg = d_sparse_scores[d_neg_idx:d_neg_idx+1]  # [1, seq_len, 1]
            d_dense_pos = d_dense_embeddings[d_pos_idx:d_pos_idx+1]  # [1, seq_len, dense_dim]
            d_dense_neg = d_dense_embeddings[d_neg_idx:d_neg_idx+1]  # [1, seq_len, dense_dim]
            d_dense_mask_pos = doc_dense_masks[d_pos_idx:d_pos_idx+1]  # [1, seq_len]
            d_dense_mask_neg = doc_dense_masks[d_neg_idx:d_neg_idx+1]  # [1, seq_len]
            
            # Compute sparse and dense scores WITHOUT accessing combination weights
            # (to avoid DDP "marked as ready twice" error)
            
            # Sparse score for positive: sum of masked impact scores
            sparse_pos_raw = (q_sparse_mask * d_sparse_pos).sum(dim=1).squeeze(-1)
            # Sparse score for negative
            sparse_neg_raw = (q_sparse_mask * d_sparse_neg).sum(dim=1).squeeze(-1)
            
            # Dense score for positive: MaxSim late interaction (pass temperature!)
            dense_pos_raw = self.model.module.compute_late_interaction_score(
                q_dense, d_dense_pos, q_dense_mask, d_dense_mask_pos, temperature
            )
            # Dense score for negative
            dense_neg_raw = self.model.module.compute_late_interaction_score(
                q_dense, d_dense_neg, q_dense_mask, d_dense_mask_neg, temperature
            )
            
            # Apply precomputed weights (computed once before loop)
            sparse_pos = w_sparse * sparse_pos_raw
            sparse_neg = w_sparse * sparse_neg_raw
            dense_pos = w_dense * dense_pos_raw
            dense_neg = w_dense * dense_neg_raw
            combined_pos = sparse_pos + dense_pos
            combined_neg = sparse_neg + dense_neg
            
            # Stack positive and negative scores
            sparse_scores_list.append(torch.stack([sparse_pos[0], sparse_neg[0]]))
            dense_scores_list.append(torch.stack([dense_pos[0], dense_neg[0]]))
            combined_scores_list.append(torch.stack([combined_pos[0], combined_neg[0]]))
        
        # Stack all scores: [batch_size, 2] where [:, 0] is positive, [:, 1] is negative
        sparse_scores = torch.stack(sparse_scores_list, dim=0)
        dense_scores = torch.stack(dense_scores_list, dim=0)
        combined_scores = torch.stack(combined_scores_list, dim=0)
        
        return sparse_scores, dense_scores, combined_scores
    
    def evaluate_loss(self, outputs, batch):
        """
        Compute combined InfoNCE loss.
        
        Args:
            outputs: Tuple of (sparse_scores, dense_scores, combined_scores)
            batch: Batch dictionary (not used)
            
        Returns:
            Total loss (scalar)
        """
        sparse_scores, dense_scores, combined_scores = outputs
        
        # Compute combined loss
        total_loss, loss_sparse, loss_dense, loss_combined = self.criterion(
            sparse_scores, dense_scores, combined_scores
        )
        
        # Log component losses
        if self.log_components and self.gpu_id == 0:
            self.loss_stats['total'] += total_loss.item()
            self.loss_stats['sparse'] += loss_sparse.item()
            self.loss_stats['dense'] += loss_dense.item()
            self.loss_stats['combined'] += loss_combined.item()
            self.loss_stats['count'] += 1
        
        return total_loss
    
    def train(self):
        """Override train to reset and log component statistics."""
        # Reset statistics
        self.loss_stats = {
            'total': 0.0,
            'sparse': 0.0,
            'dense': 0.0,
            'combined': 0.0,
            'count': 0,
        }
        
        # Call parent train method
        super().train()
        
        # Log final statistics
        if self.gpu_id == 0 and self.loss_stats['count'] > 0:
            count = self.loss_stats['count']
            self.logger.info(
                f"\nFinal Loss Statistics:\n"
                f"  Average Total Loss: {self.loss_stats['total'] / count:.6f}\n"
                f"  Average Sparse Loss: {self.loss_stats['sparse'] / count:.6f}\n"
                f"  Average Dense Loss: {self.loss_stats['dense'] / count:.6f}\n"
                f"  Average Combined Loss: {self.loss_stats['combined'] / count:.6f}"
            )
            
            # Write to metrics file
            with open(self.checkpoint_dir / "loss_stats.json", "w") as f:
                json.dump({
                    'avg_total_loss': self.loss_stats['total'] / count,
                    'avg_sparse_loss': self.loss_stats['sparse'] / count,
                    'avg_dense_loss': self.loss_stats['dense'] / count,
                    'avg_combined_loss': self.loss_stats['combined'] / count,
                }, f, indent=2)


def meta_embed_collate_fn(batch, model_cls, max_length=None):
    """
    Collate function for MetaEmbed training with sparse and dense components.
    
    Creates separate masks for:
    - Sparse scoring: query terms in documents
    - Dense scoring: expansion tokens in queries and documents
    
    Args:
        batch: List of (query, positive_doc, negative_doc) tuples
        model_cls: MetaEmbedDeepImpact class
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with:
            - query_encoded_list: Encoded queries
            - encoded_list: Encoded documents (positive, negative pairs)
            - query_sparse_masks: Masks for sparse scoring
            - query_dense_masks: Masks for query expansion tokens
            - doc_dense_masks: Masks for document expansion tokens
    """
    query_encoded_list = []
    encoded_list = []
    query_sparse_masks = []
    query_dense_masks = []
    doc_dense_masks = []
    
    for query, positive_document, negative_document in batch:
        # Process query
        query_terms = model_cls.process_query(query)
        query_encoded, query_term_to_token_index = model_cls.process_document(query)
        
        # Query expansion token mask (for dense component)
        query_exp_mask = model_cls.get_expansion_token_mask(query_term_to_token_index, max_length)
        
        # Process positive document
        pos_encoded, pos_term_to_token_index = model_cls.process_document(positive_document)
        pos_sparse_mask = model_cls.get_query_document_token_mask(query_terms, pos_term_to_token_index, max_length)
        pos_exp_mask = model_cls.get_expansion_token_mask(pos_term_to_token_index, max_length)
        
        # Process negative document
        neg_encoded, neg_term_to_token_index = model_cls.process_document(negative_document)
        neg_sparse_mask = model_cls.get_query_document_token_mask(query_terms, neg_term_to_token_index, max_length)
        neg_exp_mask = model_cls.get_expansion_token_mask(neg_term_to_token_index, max_length)
        
        # Add to lists
        query_encoded_list.append(query_encoded)
        query_dense_masks.append(query_exp_mask)
        
        # For each query, we have positive and negative documents
        encoded_list.extend([pos_encoded, neg_encoded])
        query_sparse_masks.extend([pos_sparse_mask, neg_sparse_mask])
        doc_dense_masks.extend([pos_exp_mask, neg_exp_mask])
    
    return {
        'query_encoded_list': query_encoded_list,
        'encoded_list': encoded_list,
        'query_sparse_masks': torch.stack(query_sparse_masks, dim=0).unsqueeze(-1),  # [batch*2, seq_len, 1]
        'query_dense_masks': torch.stack(query_dense_masks, dim=0).unsqueeze(-1),  # [batch, seq_len, 1]
        'doc_dense_masks': torch.stack(doc_dense_masks, dim=0).unsqueeze(-1),  # [batch*2, seq_len, 1]
    }


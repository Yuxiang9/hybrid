"""
Hybrid Distillation Trainer for Dual-Scoring Deep Impact Model.

This trainer implements knowledge distillation with integrated scoring:
- s1: Regular Deep Impact scores (weighted)
- s2: Expansion token scores (weighted)
- s: Integrated score = s1 + s2
- Loss: Distillation loss using integrated score s against teacher scores
"""

import torch
from typing import Union
from pathlib import Path

from .distil_trainer import DistilTrainer, DistilKLLoss
from ..models.hybrid_impact import HybridDeepImpact


class HybridDistilTrainer(DistilTrainer):
    """
    Trainer for Hybrid Deep Impact model with dual scoring and knowledge distillation.
    
    The trainer:
    1. Computes s1 (regular term scores) and s2 (expansion token scores) separately
    2. Combines them into integrated score s
    3. Applies distillation loss (KL divergence or MSE) using the integrated score
    
    This allows the model to learn both:
    - Traditional term-based scoring (via s1)
    - Expansion token scoring for document synthesis (via s2)
    """
    
    loss = DistilKLLoss()
    
    def __init__(
        self,
        model: HybridDeepImpact,
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
        log_score_components: bool = True,
    ):
        """
        Initialize Hybrid Distillation Trainer.
        
        Args:
            model: HybridDeepImpact model
            optimizer: Optimizer for training
            train_data: Training data loader
            checkpoint_dir: Directory for saving checkpoints
            batch_size: Batch size
            save_every: Save checkpoint every N steps
            save_best: Whether to save best model
            seed: Random seed
            gradient_accumulation_steps: Number of gradient accumulation steps
            eval_every: Evaluate every N steps
            evaluator: Evaluator for validation
            log_score_components: Whether to log s1, s2, s separately for analysis
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
        
        self.log_score_components = log_score_components
        self.score_stats = {
            's1_sum': 0.0,
            's2_sum': 0.0,
            's_sum': 0.0,
            'w1_sum': 0.0,
            'w2_sum': 0.0,
            'count': 0,
        }
    
    def get_output_scores(self, batch):
        """
        Compute integrated scores (s) from regular (s1) and expansion (s2) scores.
        Uses late interaction maxsim for s2 expansion token scoring.
        
        This method:
        1. Encodes queries to get query expansion embeddings
        2. Encodes documents to get regular scores and document expansion embeddings
        3. Computes s1 (regular term matching) and s2 (late interaction maxsim)
        4. Combines them using learned weights
        
        Args:
            batch: Dictionary containing:
                - query_encoded_list: List of encoded queries
                - encoded_list: List of encoded documents
                - regular_masks: Masks for regular terms [batch*2, seq_len, 1]
                - query_expansion_masks: Masks for query expansion tokens [batch*2, seq_len]
                - doc_expansion_masks: Masks for document expansion tokens [batch*2, seq_len]
                
        Returns:
            Integrated scores [batch_size, 2] where each row is [pos_score, neg_score]
        """
        # Encode queries to get query expansion embeddings
        q_input_ids, q_attention_mask, q_type_ids = self.get_input_tensors(batch['query_encoded_list'])
        
        _, query_expansion_embs, _ = self.model(
            q_input_ids,
            q_attention_mask,
            q_type_ids,
            return_expansion_embeddings=True
        )
        
        # Encode documents to get regular scores and document expansion embeddings
        d_input_ids, d_attention_mask, d_type_ids = self.get_input_tensors(batch['encoded_list'])
        
        regular_scores, doc_expansion_embs, _ = self.model(
            d_input_ids,
            d_attention_mask,
            d_type_ids,
            return_expansion_embeddings=True
        )
        
        # Get masks
        regular_masks = batch['regular_masks'].to(self.gpu_id)
        query_expansion_masks = batch['query_expansion_masks'].to(self.gpu_id).squeeze(-1)  # [batch*2, seq_len]
        doc_expansion_masks = batch['doc_expansion_masks'].to(self.gpu_id).squeeze(-1)  # [batch*2, seq_len]
        
        # Combine scores using the model's method with late interaction maxsim
        s1, s2, s = self.model.module.get_combined_scores(
            regular_scores,
            query_expansion_embs,
            doc_expansion_embs,
            regular_masks,
            query_expansion_masks,
            doc_expansion_masks
        )
        
        # Reshape to [batch_size, 2] format (positive, negative)
        s_integrated = s.view(self.batch_size, -1)
        
        # Log score components for analysis (optional)
        if self.log_score_components and self.gpu_id == 0:
            with torch.no_grad():
                s1_reshaped = s1.view(self.batch_size, -1)
                s2_reshaped = s2.view(self.batch_size, -1)
                
                # Get normalized weights
                weights = torch.softmax(
                    torch.stack([self.model.module.w1, self.model.module.w2]),
                    dim=0
                )
                
                self.score_stats['s1_sum'] += s1_reshaped.mean().item()
                self.score_stats['s2_sum'] += s2_reshaped.mean().item()
                self.score_stats['s_sum'] += s_integrated.mean().item()
                self.score_stats['w1_sum'] += weights[0].item()
                self.score_stats['w2_sum'] += weights[1].item()
                self.score_stats['count'] += 1
        
        return s_integrated
    
    def train(self):
        """Override train to add score component logging."""
        # Reset stats
        self.score_stats = {
            's1_sum': 0.0,
            's2_sum': 0.0,
            's_sum': 0.0,
            'w1_sum': 0.0,
            'w2_sum': 0.0,
            'count': 0,
        }
        
        # Call parent train
        super().train()
        
        # Log final statistics
        if self.gpu_id == 0 and self.score_stats['count'] > 0:
            count = self.score_stats['count']
            self.logger.info(
                f"\nFinal Score Statistics:\n"
                f"  Average s1 (regular): {self.score_stats['s1_sum'] / count:.4f}\n"
                f"  Average s2 (expansion): {self.score_stats['s2_sum'] / count:.4f}\n"
                f"  Average s (integrated): {self.score_stats['s_sum'] / count:.4f}\n"
                f"  Average w1: {self.score_stats['w1_sum'] / count:.4f}\n"
                f"  Average w2: {self.score_stats['w2_sum'] / count:.4f}"
            )
    
    def get_score_statistics(self):
        """Get current score statistics for logging during training."""
        if self.score_stats['count'] == 0:
            return {}
        
        count = self.score_stats['count']
        return {
            'avg_s1': self.score_stats['s1_sum'] / count,
            'avg_s2': self.score_stats['s2_sum'] / count,
            'avg_s': self.score_stats['s_sum'] / count,
            'avg_w1': self.score_stats['w1_sum'] / count,
            'avg_w2': self.score_stats['w2_sum'] / count,
        }


def hybrid_distil_collate_fn(batch, model_cls=HybridDeepImpact, max_length=None):
    """
    Collate function for hybrid distillation training with late interaction.
    
    Creates:
    - Query encodings with expansion token masks
    - Document encodings with regular term masks and expansion token masks
    
    Args:
        batch: List of (query, [(passage, score), ...]) tuples
        model_cls: Model class (HybridDeepImpact)
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with:
            - query_encoded_list: List of encoded queries (one per document)
            - encoded_list: List of encoded documents
            - regular_masks: Masks for regular query terms in documents
            - query_expansion_masks: Masks for expansion tokens in queries
            - doc_expansion_masks: Masks for expansion tokens in documents
            - scores: Teacher scores for distillation
    """
    query_encoded_list, encoded_list = [], []
    regular_masks, query_expansion_masks, doc_expansion_masks = [], [], []
    scores = []
    
    for query, pid_score_list in batch:
        # Encode query once
        query_encoded, query_term_to_token_index = model_cls.process_document(query)
        query_exp_mask = model_cls.get_expansion_token_mask(query_term_to_token_index, max_length)
        
        # Process each document
        for passage, score in pid_score_list:
            # Process document with query for regular mask
            encoded_token, regular_mask, doc_exp_mask = model_cls.process_query_and_document_hybrid(
                query, passage, max_length=max_length
            )
            
            # Add query encoding (repeated for each document)
            query_encoded_list.append(query_encoded)
            query_expansion_masks.append(query_exp_mask)
            
            # Add document encoding
            encoded_list.append(encoded_token)
            regular_masks.append(regular_mask)
            doc_expansion_masks.append(doc_exp_mask)
            
            scores.append(score)
    
    return {
        'query_encoded_list': query_encoded_list,
        'encoded_list': encoded_list,
        'regular_masks': torch.stack(regular_masks, dim=0).unsqueeze(-1),
        'query_expansion_masks': torch.stack(query_expansion_masks, dim=0).unsqueeze(-1),
        'doc_expansion_masks': torch.stack(doc_expansion_masks, dim=0).unsqueeze(-1),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


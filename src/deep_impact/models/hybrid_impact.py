"""
Hybrid Deep Impact Model with Dual Scoring Mechanisms.

This model implements a hybrid approach inspired by meta-token expansion:
- s1: Regular Deep Impact scores for document terms
- s2: Expansion token scores (meta tokens for text synthesis)
- s: Integrated score = w1*s1 + w2*s2

The dual scoring allows the model to leverage both:
1. Traditional term-based scoring (Deep Impact)
2. Learned expansion token scoring (similar to meta tokens in literature)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Set, List, Optional
from .original import DeepImpact


class HybridDeepImpact(DeepImpact):
    """
    Hybrid Deep Impact model with separate scoring heads for:
    1. Regular document terms (s1)
    2. Expansion tokens (s2) - special tokens for document synthesis
    
    The final score is a weighted combination: s = w1*s1 + w2*s2
    """
    
    def __init__(self, config, expansion_weight: float = 0.3, regular_weight: float = 0.7):
        """
        Initialize the Hybrid Deep Impact model.
        
        Args:
            config: BERT configuration
            expansion_weight (w2): Weight for expansion token scores (default: 0.3)
            regular_weight (w1): Weight for regular term scores (default: 0.7)
        """
        # Don't call DeepImpact.__init__ directly to avoid duplicate initialization
        # Instead, call BertPreTrainedModel.__init__
        from transformers import BertPreTrainedModel, BertModel
        BertPreTrainedModel.__init__(self, config)
        
        self.bert = BertModel(config)
        
        # Dual scoring heads
        self.regular_impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )
        
        self.expansion_impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.ReLU()
        )
        
        # Learnable weights for combining scores
        # Initialize as parameters so they can be learned during training
        self.w1 = nn.Parameter(torch.tensor(regular_weight))
        self.w2 = nn.Parameter(torch.tensor(expansion_weight))
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        return_expansion_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual scoring.
        
        Args:
            input_ids: Batch of input ids
            attention_mask: Batch of attention masks
            token_type_ids: Batch of token type ids
            return_expansion_embeddings: If True, return expansion embeddings for late interaction
            
        Returns:
            (regular_scores, expansion_output, last_hidden_state) where:
                - regular_scores: Scalar scores for regular terms [batch, seq_len, 1]
                - expansion_output: Embeddings [batch, seq_len, hidden_dim] if return_expansion_embeddings=True,
                                    otherwise scalar scores [batch, seq_len, 1]
                - last_hidden_state: BERT hidden states [batch, seq_len, hidden_dim]
        """
        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_output.last_hidden_state
        
        # Regular scoring: scalar scores for exact term matching
        regular_scores = self.regular_impact_score_encoder(last_hidden_state)
        
        # Expansion scoring: return embeddings for late interaction maxsim
        if return_expansion_embeddings:
            # Return normalized embeddings for late interaction
            # Use the intermediate hidden layer before final projection
            expansion_embeddings = self.expansion_impact_score_encoder[0](last_hidden_state)  # First Linear layer
            expansion_embeddings = self.expansion_impact_score_encoder[1](expansion_embeddings)  # ReLU
            # L2 normalize for dot product similarity
            expansion_embeddings = torch.nn.functional.normalize(expansion_embeddings, p=2, dim=-1)
            return regular_scores, expansion_embeddings, last_hidden_state
        else:
            # Return scalar scores (for inference/compatibility)
            expansion_scores = self.expansion_impact_score_encoder(last_hidden_state)
            return regular_scores, expansion_scores, last_hidden_state
    
    def get_combined_scores(
        self,
        regular_scores: torch.Tensor,
        query_expansion_embs: torch.Tensor,
        doc_expansion_embs: torch.Tensor,
        regular_mask: torch.Tensor,
        query_expansion_mask: torch.Tensor,
        doc_expansion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine regular and expansion scores with learned weights.
        Uses late interaction maxsim for expansion token scoring.
        
        Args:
            regular_scores: Scores for regular terms [batch, seq_len, 1]
            query_expansion_embs: Query expansion embeddings [batch, seq_len, hidden_dim]
            doc_expansion_embs: Document expansion embeddings [batch, seq_len, hidden_dim]
            regular_mask: Mask for regular terms [batch, seq_len, 1]
            query_expansion_mask: Mask for query expansion tokens [batch, seq_len]
            doc_expansion_mask: Mask for document expansion tokens [batch, seq_len]
            
        Returns:
            Tuple of (s1, s2, s) where:
                s1: Weighted regular term scores
                s2: Weighted expansion token scores (late interaction maxsim)
                s: Integrated score (s1 + s2)
        """
        # S1: Regular term scores (Deep Impact style - sum of masked scores)
        s1_raw = (regular_mask * regular_scores).sum(dim=1).squeeze(-1)
        
        # S2: Late interaction maxsim for expansion tokens
        # For each query expansion token, find max similarity with document expansion tokens
        batch_size = query_expansion_embs.shape[0]
        s2_raw = torch.zeros(batch_size, device=query_expansion_embs.device)
        
        for i in range(batch_size):
            # Get valid query expansion tokens (where mask is True)
            q_mask = query_expansion_mask[i]  # [seq_len]
            q_embs = query_expansion_embs[i][q_mask]  # [num_query_exp, hidden_dim]
            
            # Get valid document expansion tokens (where mask is True)
            d_mask = doc_expansion_mask[i]  # [seq_len]
            d_embs = doc_expansion_embs[i][d_mask]  # [num_doc_exp, hidden_dim]
            
            if q_embs.shape[0] > 0 and d_embs.shape[0] > 0:
                # Compute similarity matrix: [num_query_exp, num_doc_exp]
                sim_matrix = torch.matmul(q_embs, d_embs.t())
                
                # For each query expansion, take max over document expansions
                max_sims = sim_matrix.max(dim=1)[0]  # [num_query_exp]
                
                # Sum across query expansions (late interaction formula)
                s2_raw[i] = max_sims.sum()
        
        # Apply learned weights (ensuring they're positive with softmax normalization)
        # This ensures w1 + w2 = 1 for stable training
        weights = torch.softmax(torch.stack([self.w1, self.w2]), dim=0)
        w1_normalized, w2_normalized = weights[0], weights[1]
        
        s1 = w1_normalized * s1_raw
        s2 = w2_normalized * s2_raw
        s = s1 + s2
        
        return s1, s2, s
    
    def _get_term_impact_scores(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Override to return combined scores for compatibility.
        This is used during inference when the model is called without explicit mask handling.
        """
        # For inference/compatibility, we'll return the regular scores
        # The hybrid scoring is primarily for training with the custom trainer
        return self.regular_impact_score_encoder(last_hidden_state)
    
    @classmethod
    def get_expansion_token_mask(
        cls,
        term_to_token_index: Dict[str, int],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create a mask specifically for expansion tokens.
        
        Args:
            term_to_token_index: Mapping from terms to token indices
            max_length: Maximum sequence length
            
        Returns:
            Boolean mask with True for expansion token positions
        """
        if max_length is None:
            max_length = cls.max_length
        
        import numpy as np
        mask = np.zeros(max_length, dtype=bool)
        
        # Mark only expansion tokens
        expansion_token_indices = [
            v for k, v in term_to_token_index.items()
            if k in cls.expansion_tokens
        ]
        mask[expansion_token_indices] = True
        
        return torch.from_numpy(mask)
    
    @classmethod
    def get_regular_term_mask(
        cls,
        query_terms: Set[str],
        term_to_token_index: Dict[str, int],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create a mask for regular (non-expansion) terms that match the query.
        
        Args:
            query_terms: Set of query terms
            term_to_token_index: Mapping from terms to token indices
            max_length: Maximum sequence length
            
        Returns:
            Boolean mask with True for regular matching term positions
        """
        if max_length is None:
            max_length = cls.max_length
        
        import numpy as np
        mask = np.zeros(max_length, dtype=bool)
        
        # Mark only regular (non-expansion) query terms
        regular_token_indices = [
            v for k, v in term_to_token_index.items()
            if (k in query_terms) and (k not in cls.expansion_tokens)
        ]
        mask[regular_token_indices] = True
        
        return torch.from_numpy(mask)
    
    @classmethod
    def process_query_and_document_hybrid(
        cls,
        query: str,
        document: str,
        max_length: Optional[int] = None
    ) -> Tuple[object, torch.Tensor, torch.Tensor]:
        """
        Process query and document for hybrid scoring with separate masks.
        
        Args:
            query: Query string
            document: Document string
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (encoded_document, regular_mask, expansion_mask)
        """
        query_terms = cls.process_query(query)
        encoded, term_to_token_index = cls.process_document(document)
        
        regular_mask = cls.get_regular_term_mask(query_terms, term_to_token_index, max_length)
        expansion_mask = cls.get_expansion_token_mask(term_to_token_index, max_length)
        
        return encoded, regular_mask, expansion_mask


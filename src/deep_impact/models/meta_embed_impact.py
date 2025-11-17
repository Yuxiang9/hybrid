"""
MetaEmbed Deep Impact: Hybrid Sparse-Dense Retrieval with Late Interaction

This model implements a hybrid approach inspired by the MetaEmbed paper:
- Sparse component: Traditional DeepImpact impact scores for lexical matching
- Dense component: Special expansion tokens with dense embeddings for late interaction (MaxSim)
- Joint training: Both components trained together with contrastive learning

The key difference from standard DeepImpact:
1. Expansion tokens produce dense embeddings (not scalar scores)
2. Late interaction scoring using MaxSim (like ColBERT) on expansion tokens
3. Combined sparse + dense score for final ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Set, List, Optional
from .original import DeepImpact


class MetaEmbedDeepImpact(DeepImpact):
    """
    Hybrid sparse-dense retrieval model combining:
    1. Sparse impact scores for document terms (lexical matching)
    2. Dense late-interaction for expansion tokens (semantic matching)
    
    Architecture:
    - BERT backbone (shared)
    - Sparse head: Linear + ReLU → scalar impact scores for all tokens
    - Dense head: Linear projection → dense embeddings ONLY for expansion tokens
    
    Scoring:
    - Sparse score: sum of impact scores for matching query terms
    - Dense score: MaxSim late interaction on expansion token embeddings
    - Final score: weighted combination of sparse + dense
    """
    
    # MetaEmbed can use different number of expansion tokens (default: None, will be set dynamically)
    # This overrides the parent's default
    _default_num_expansion_tokens = None  # Will be set via set_expansion_tokens()
    
    @classmethod
    def set_expansion_tokens(cls, num_tokens: int):
        """
        Set the number of expansion tokens for MetaEmbedDeepImpact.
        
        Unlike the parent DeepImpact (which uses 32 by default), MetaEmbed
        can use any number. This must be called before creating model instances.
        
        Args:
            num_tokens: Number of expansion tokens to use
        """
        # Update the default for this class
        cls._default_num_expansion_tokens = num_tokens
        
        # Get current count (may be None if not initialized yet)
        old_count = len(cls.expansion_tokens) if cls.expansion_tokens is not None else 0
        
        # Set new expansion tokens
        cls.expansion_tokens = [f"exp{i}" for i in range(num_tokens)]
        
        # Add new tokens to tokenizer if expanding beyond current count
        if num_tokens > old_count:
            new_tokens = [f"exp{i}" for i in range(old_count, num_tokens)]
            cls.tokenizer.add_tokens(new_tokens)
            print(f"MetaEmbedDeepImpact: Added {len(new_tokens)} expansion tokens (total: {num_tokens})")
        elif num_tokens < old_count and old_count > 0:
            print(f"Warning: Reducing expansion tokens from {old_count} to {num_tokens}")
            print(f"Note: Tokenizer still has all {old_count} tokens, but only first {num_tokens} will be used")
    
    def __init__(self, config, dense_dim: int = 128, num_expansion_tokens: int = 64, 
                 sparse_weight: float = 0.5, dense_weight: float = 0.5,
                 learn_weights: bool = True):
        """
        Initialize MetaEmbed Deep Impact model.
        
        Args:
            config: BERT configuration
            dense_dim: Dimensionality of dense embeddings for expansion tokens (default: 128)
            num_expansion_tokens: Number of special expansion tokens to add (default: 64)
            sparse_weight: Weight for sparse component (default: 0.5)
            dense_weight: Weight for dense component (default: 0.5)
            learn_weights: If True, weights are learnable parameters. If False, fixed. (default: True)
        """
        # First, update the class-level expansion tokens before calling super().__init__
        # This ensures the tokenizer has the correct number of expansion tokens
        # Note: set_expansion_tokens should have been called before __init__ in train.py
        # But we update the list here for consistency
        self.__class__.expansion_tokens = [f"exp{i}" for i in range(num_expansion_tokens)]
        
        # Initialize parent (this will call BertPreTrainedModel.__init__)
        super().__init__(config)
        
        # Store configuration
        self.dense_dim = dense_dim
        self.num_expansion_tokens = num_expansion_tokens
        self.learn_weights = learn_weights
        
        # Sparse head: produces scalar impact scores for ALL tokens (standard DeepImpact)
        # This is already initialized by parent as self.impact_score_encoder
        
        # Dense head: produces dense embeddings ONLY for expansion tokens
        # We'll extract these embeddings and use them for late interaction
        self.dense_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, dense_dim),
        )
        
        # Combination weights: learnable OR fixed
        if learn_weights:
            # Learnable weights (trained via gradient descent)
            # Using log space for numerical stability during training
            self.log_sparse_weight = nn.Parameter(torch.tensor(sparse_weight).log())
            self.log_dense_weight = nn.Parameter(torch.tensor(dense_weight).log())
        else:
            # Fixed weights (not trainable) - avoids DDP issues entirely!
            # Register as buffers so they're part of state_dict but not parameters
            self.register_buffer('log_sparse_weight', torch.tensor(sparse_weight).log())
            self.register_buffer('log_dense_weight', torch.tensor(dense_weight).log())
        
        # Temperature parameter for late interaction (like ColBERT)
        # This scales the similarity scores
        # Always learnable for now (can make this optional too if needed)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        return_dense_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass producing both sparse scores and (optionally) dense embeddings.
        
        Args:
            input_ids: Batch of input ids [batch_size, seq_len]
            attention_mask: Batch of attention masks [batch_size, seq_len]
            token_type_ids: Batch of token type ids [batch_size, seq_len]
            return_dense_embeddings: If True, return dense embeddings for expansion tokens
            
        Returns:
            - sparse_scores: Impact scores for all tokens [batch_size, seq_len, 1]
            - dense_embeddings: (if return_dense_embeddings=True) 
                               Dense embeddings [batch_size, seq_len, dense_dim]
        """
        # Get BERT representations
        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Sparse scores: apply impact score encoder to all tokens
        sparse_scores = self.impact_score_encoder(last_hidden_state)  # [batch_size, seq_len, 1]
        
        # Dense embeddings: apply projection and normalize for dot product similarity
        if return_dense_embeddings:
            dense_embeddings = self.dense_projection(last_hidden_state)  # [batch_size, seq_len, dense_dim]
            # L2 normalize for efficient dot product similarity computation
            dense_embeddings = F.normalize(dense_embeddings, p=2, dim=-1)
            return sparse_scores, dense_embeddings
        
        return sparse_scores, None
    
    def compute_late_interaction_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim late interaction score between query and document expansion tokens.
        
        This implements the ColBERT-style late interaction:
        - For each query token embedding, find max similarity with all doc token embeddings
        - Sum these max similarities across all query tokens
        
        Args:
            query_embeddings: Query embeddings [batch_size, seq_len_q, dense_dim]
            doc_embeddings: Document embeddings [batch_size, seq_len_d, dense_dim]
            query_mask: Mask for valid query expansion tokens [batch_size, seq_len_q]
            doc_mask: Mask for valid document expansion tokens [batch_size, seq_len_d]
            temperature: Optional temperature value (if None, uses self.temperature)
            
        Returns:
            Late interaction scores [batch_size]
        """
        # Use provided temperature or default to self.temperature
        if temperature is None:
            temperature = self.temperature
            
        batch_size = query_embeddings.shape[0]
        scores = torch.zeros(batch_size, device=query_embeddings.device)
        
        # Convert masks to boolean for indexing (they may be float from collate_fn)
        query_mask_bool = query_mask.bool()
        doc_mask_bool = doc_mask.bool()
        
        for i in range(batch_size):
            # Get valid embeddings using masks
            q_valid = query_embeddings[i][query_mask_bool[i]]  # [num_query_tokens, dense_dim]
            d_valid = doc_embeddings[i][doc_mask_bool[i]]      # [num_doc_tokens, dense_dim]
            
            if q_valid.shape[0] > 0 and d_valid.shape[0] > 0:
                # Compute similarity matrix: [num_query_tokens, num_doc_tokens]
                # Since embeddings are L2-normalized, dot product = cosine similarity
                similarity = torch.matmul(q_valid, d_valid.t())  
                
                # Apply temperature scaling (using passed value, not parameter!)
                similarity = similarity / temperature
                
                # MaxSim: for each query token, take max similarity with doc tokens
                max_similarities = similarity.max(dim=1)[0]  # [num_query_tokens]
                
                # Sum across query tokens
                scores[i] = max_similarities.sum()
        
        return scores
    
    def get_combined_scores(
        self,
        query_sparse_scores: torch.Tensor,
        doc_sparse_scores: torch.Tensor,
        query_dense_embeddings: torch.Tensor,
        doc_dense_embeddings: torch.Tensor,
        query_sparse_mask: torch.Tensor,
        doc_sparse_mask: torch.Tensor,
        query_dense_mask: torch.Tensor,
        doc_dense_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine sparse and dense scores with learned weights.
        
        Args:
            query_sparse_scores: Query sparse scores [batch_size, seq_len, 1]
            doc_sparse_scores: Document sparse scores [batch_size, seq_len, 1]
            query_dense_embeddings: Query dense embeddings [batch_size, seq_len, dense_dim]
            doc_dense_embeddings: Document dense embeddings [batch_size, seq_len, dense_dim]
            query_sparse_mask: Mask for query terms in document [batch_size, seq_len, 1]
            doc_sparse_mask: Not used (keeping for interface compatibility)
            query_dense_mask: Mask for query expansion tokens [batch_size, seq_len]
            doc_dense_mask: Mask for document expansion tokens [batch_size, seq_len]
            
        Returns:
            Tuple of (sparse_score, dense_score, combined_score) each [batch_size]
        """
        # Sparse score: sum of impact scores for matching query terms in document
        # This is the standard DeepImpact scoring
        sparse_score = (query_sparse_mask * doc_sparse_scores).sum(dim=1).squeeze(-1)
        
        # Dense score: MaxSim late interaction on expansion tokens
        dense_score = self.compute_late_interaction_score(
            query_dense_embeddings,
            doc_dense_embeddings,
            query_dense_mask,
            doc_dense_mask,
        )
        
        # Combine with learned weights (using softmax for normalization)
        weights = F.softmax(torch.stack([self.log_sparse_weight, self.log_dense_weight]), dim=0)
        sparse_weight = weights[0]
        dense_weight = weights[1]
        
        combined_score = sparse_weight * sparse_score + dense_weight * dense_score
        
        return sparse_score, dense_score, combined_score
    
    @classmethod
    def load(cls, checkpoint_path: Optional[str] = None, dense_dim: int = 128, 
             num_expansion_tokens: int = 64, learn_weights: bool = True):
        """
        Load model with proper initialization of expansion tokens.
        
        Args:
            checkpoint_path: Path to checkpoint (if None, loads pretrained base)
            dense_dim: Dimensionality of dense embeddings
            num_expansion_tokens: Number of expansion tokens to use
            learn_weights: If True, weights are learnable. If False, fixed.
            
        Returns:
            Loaded model
        """
        from transformers import AutoConfig
        import os
        from src.utils.checkpoint import ModelCheckpoint
        
        # Update class-level expansion tokens BEFORE creating model
        # Use the parent class method to properly update tokenizer
        cls.set_expansion_tokens(num_expansion_tokens)
        
        # Load configuration
        config = AutoConfig.from_pretrained('Luyu/co-condenser-marco')
        
        # Create model
        model = cls(config, dense_dim=dense_dim, num_expansion_tokens=num_expansion_tokens,
                   learn_weights=learn_weights)
        
        # Resize token embeddings to account for new expansion tokens
        cls.tokenizer.enable_truncation(max_length=cls.max_length, strategy='longest_first')
        cls.tokenizer.enable_padding(length=cls.max_length)
        vocab_size = cls.tokenizer.get_vocab_size()
        if vocab_size != model.bert.embeddings.word_embeddings.num_embeddings:
            model.resize_token_embeddings(vocab_size)
        
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
        
        return model
    
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
        
        # Mark only expansion token positions
        expansion_indices = [
            idx for term, idx in term_to_token_index.items()
            if term in cls.expansion_tokens
        ]
        mask[expansion_indices] = True
        
        return torch.from_numpy(mask)
    
    def get_impact_scores_batch(self, documents: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Get impact scores for inference (uses only sparse component for efficiency).
        For full sparse+dense retrieval, use the appropriate retrieval pipeline.
        
        Args:
            documents: List of document strings
            
        Returns:
            List of lists of (term, impact_score) tuples
        """
        # For inference, we only use sparse scores (dense requires query)
        encoded_docs = []
        term_to_token_maps = []
        for doc in documents:
            encoded, term_map = self.process_document(doc)
            encoded_docs.append(encoded)
            term_to_token_maps.append(term_map)

        # Create batched tensors
        input_ids = torch.tensor([enc.ids for enc in encoded_docs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([enc.attention_mask for enc in encoded_docs], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([enc.type_ids for enc in encoded_docs], dtype=torch.long).to(self.device)

        # Get sparse scores only
        with torch.no_grad():
            sparse_scores, _ = self(input_ids, attention_mask, token_type_ids, return_dense_embeddings=False)

        # Compute impact scores for all documents
        return self.compute_term_impacts(term_to_token_maps, sparse_scores)


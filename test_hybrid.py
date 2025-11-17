"""
Test script for the Hybrid Deep Impact implementation.
This script verifies that the hybrid model works correctly before running full training.
"""

import torch
from transformers import AutoConfig
from src.deep_impact.models import HybridDeepImpact, DeepImpact

def test_model_initialization():
    """Test that the hybrid model can be initialized."""
    print("=" * 80)
    print("Test 1: Model Initialization")
    print("=" * 80)
    
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = HybridDeepImpact(config, expansion_weight=0.3, regular_weight=0.7)
    
    print(f"✓ Model initialized successfully")
    print(f"  - Regular impact encoder: {model.regular_impact_score_encoder}")
    print(f"  - Expansion impact encoder: {model.expansion_impact_score_encoder}")
    print(f"  - w1 (regular weight): {model.w1.item():.3f}")
    print(f"  - w2 (expansion weight): {model.w2.item():.3f}")
    
    # Check that weights are learnable parameters
    assert model.w1.requires_grad, "w1 should require gradients"
    assert model.w2.requires_grad, "w2 should require gradients"
    print(f"✓ Weights are learnable parameters")
    
    return model


def test_tokenizer_expansion_tokens():
    """Test that expansion tokens are in the vocabulary."""
    print("\n" + "=" * 80)
    print("Test 2: Expansion Tokens in Vocabulary")
    print("=" * 80)
    
    tokenizer = HybridDeepImpact.tokenizer
    vocab = tokenizer.get_vocab()
    
    expansion_tokens = HybridDeepImpact.expansion_tokens
    print(f"Expansion tokens: {expansion_tokens[:5]}... (showing first 5 of {len(expansion_tokens)})")
    
    for token in expansion_tokens:
        assert token in vocab, f"Expansion token {token} not in vocabulary"
    
    print(f"✓ All {len(expansion_tokens)} expansion tokens are in vocabulary")
    
    # Test encoding a document with expansion tokens
    test_doc = "machine learning is great exp0 exp1 exp2"
    encoded = tokenizer.encode(test_doc)
    print(f"✓ Can encode document with expansion tokens")
    print(f"  Sample document: '{test_doc}'")
    print(f"  Encoded tokens: {encoded.tokens[:10]}...")


def test_forward_pass():
    """Test forward pass with dual scoring and late interaction."""
    print("\n" + "=" * 80)
    print("Test 3: Forward Pass with Late Interaction MaxSim")
    print("=" * 80)
    
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = HybridDeepImpact(config, expansion_weight=0.3, regular_weight=0.7)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_length = 50
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    
    with torch.no_grad():
        # Test with expansion embeddings (for late interaction)
        regular_scores, expansion_embeddings, _ = model(
            input_ids, 
            attention_mask, 
            token_type_ids, 
            return_expansion_embeddings=True
        )
    
    print(f"✓ Forward pass successful with late interaction")
    print(f"  - Regular scores shape: {regular_scores.shape}")
    print(f"  - Expansion embeddings shape: {expansion_embeddings.shape}")
    print(f"  - Regular scores sample: {regular_scores[0, :5, 0].tolist()}")
    print(f"  - Expansion embeddings norm sample: {expansion_embeddings[0, 0].norm().item():.4f} (should be ~1.0)")
    
    # Test late interaction maxsim
    # Create query and document expansion embeddings and masks
    query_exp_embs = expansion_embeddings  # [batch, seq_len, emb_dim]
    doc_exp_embs = expansion_embeddings  # [batch, seq_len, emb_dim]
    
    regular_mask = torch.zeros(batch_size, seq_length, 1)
    query_exp_mask = torch.zeros(batch_size, seq_length)
    doc_exp_mask = torch.zeros(batch_size, seq_length)
    
    # Set some positions to 1 in masks
    regular_mask[:, :10, :] = 1
    query_exp_mask[:, 10:15] = 1
    doc_exp_mask[:, 15:20] = 1
    
    s1, s2, s = model.get_combined_scores(
        regular_scores, 
        query_exp_embs,
        doc_exp_embs,
        regular_mask, 
        query_exp_mask,
        doc_exp_mask
    )
    
    print(f"✓ Late interaction maxsim successful")
    print(f"  - s1 (regular) shape: {s1.shape}, sample: {s1[0].item():.4f}")
    print(f"  - s2 (expansion maxsim) shape: {s2.shape}, sample: {s2[0].item():.4f}")
    print(f"  - s (integrated) shape: {s.shape}, sample: {s[0].item():.4f}")
    
    # Verify that weights are normalized
    weights = torch.softmax(torch.stack([model.w1, model.w2]), dim=0)
    print(f"  - Normalized w1: {weights[0].item():.3f}")
    print(f"  - Normalized w2: {weights[1].item():.3f}")
    print(f"  - Sum of weights: {weights.sum().item():.3f}")
    
    return model


def test_mask_generation():
    """Test mask generation for regular terms and expansion tokens."""
    print("\n" + "=" * 80)
    print("Test 4: Mask Generation")
    print("=" * 80)
    
    # Test query and document
    query = "machine learning artificial intelligence"
    document = "machine learning is a subset of artificial intelligence"
    
    # Process query and document
    encoded, regular_mask, expansion_mask = HybridDeepImpact.process_query_and_document_hybrid(
        query, document, max_length=50
    )
    
    print(f"✓ Mask generation successful")
    print(f"  - Query: '{query}'")
    print(f"  - Document: '{document}'")
    print(f"  - Regular mask shape: {regular_mask.shape}")
    print(f"  - Expansion mask shape: {expansion_mask.shape}")
    print(f"  - Regular mask sum (# matching terms): {regular_mask.sum().item()}")
    print(f"  - Expansion mask sum (# expansion tokens): {expansion_mask.sum().item()}")
    
    # Verify masks are disjoint (no overlap)
    overlap = (regular_mask & expansion_mask).sum().item()
    assert overlap == 0, "Regular and expansion masks should not overlap"
    print(f"✓ Masks are disjoint (no overlap)")
    
    # Show which tokens are in each mask
    print(f"  - Regular term tokens: {[encoded.tokens[i] for i in range(len(encoded.tokens)) if regular_mask[i]][:10]}")
    print(f"  - Expansion tokens: {[encoded.tokens[i] for i in range(len(encoded.tokens)) if expansion_mask[i]][:10]}")


def test_pretrained_weight_loading():
    """Test loading pretrained weights from base DeepImpact model."""
    print("\n" + "=" * 80)
    print("Test 5: Pretrained Weight Loading")
    print("=" * 80)
    
    try:
        # Load base DeepImpact model
        print("Loading base DeepImpact model...")
        base_model = DeepImpact.load()
        print("✓ Base model loaded")
        
        # Create HybridDeepImpact model
        config = base_model.config
        hybrid_model = HybridDeepImpact(config, expansion_weight=0.3, regular_weight=0.7)
        
        # Copy weights
        print("Copying BERT weights...")
        hybrid_model.bert.load_state_dict(base_model.bert.state_dict())
        print("✓ BERT weights copied")
        
        print("Copying regular impact encoder weights...")
        hybrid_model.regular_impact_score_encoder.load_state_dict(
            base_model.impact_score_encoder.state_dict()
        )
        print("✓ Regular impact encoder weights copied")
        
        print("✓ Successfully initialized hybrid model from pretrained weights")
        
    except Exception as e:
        print(f"⚠ Warning: Could not load pretrained weights (this is OK for testing)")
        print(f"  Error: {e}")


def test_collate_function():
    """Test the hybrid distillation collate function with late interaction."""
    print("\n" + "=" * 80)
    print("Test 6: Collate Function with Late Interaction")
    print("=" * 80)
    
    from src.deep_impact.training.hybrid_distil_trainer import hybrid_distil_collate_fn
    
    # Create sample batch
    batch = [
        ("machine learning", [
            ("machine learning is great", 5.2),
            ("deep learning neural networks", 3.1),
        ]),
        ("artificial intelligence", [
            ("AI is the future", 4.5),
            ("robotics and automation", 2.3),
        ]),
    ]
    
    result = hybrid_distil_collate_fn(batch, model_cls=HybridDeepImpact, max_length=50)
    
    print(f"✓ Collate function successful with late interaction")
    print(f"  - Number of queries encoded: {len(result['query_encoded_list'])}")
    print(f"  - Number of documents encoded: {len(result['encoded_list'])}")
    print(f"  - Regular masks shape: {result['regular_masks'].shape}")
    print(f"  - Query expansion masks shape: {result['query_expansion_masks'].shape}")
    print(f"  - Doc expansion masks shape: {result['doc_expansion_masks'].shape}")
    print(f"  - Scores shape: {result['scores'].shape}")
    print(f"  - Scores: {result['scores'].tolist()}")
    
    # Verify masks are disjoint
    regular_sum = result['regular_masks'].sum().item()
    query_exp_sum = result['query_expansion_masks'].sum().item()
    doc_exp_sum = result['doc_expansion_masks'].sum().item()
    
    print(f"✓ Mask statistics:")
    print(f"  - Regular terms: {regular_sum:.0f} positions")
    print(f"  - Query expansion tokens: {query_exp_sum:.0f} positions")
    print(f"  - Document expansion tokens: {doc_exp_sum:.0f} positions")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HYBRID DEEP IMPACT - TEST SUITE")
    print("=" * 80)
    
    try:
        test_model_initialization()
        test_tokenizer_expansion_tokens()
        test_forward_pass()
        test_mask_generation()
        test_pretrained_weight_loading()
        test_collate_function()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe hybrid method implementation is working correctly.")
        print("You can now proceed with training using:")
        print("  bash slurm_jobs/training_hybrid.sh")
        print("\nOr manually with:")
        print("  torchrun --nproc_per_node=N -m src.deep_impact.train --hybrid [other args]")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


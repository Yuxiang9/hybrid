#!/usr/bin/env python
"""
Smoke test for MetaEmbed-KD training.
Tests model initialization, forward pass, collate function, and a mini training step.
Run this to verify everything works before launching full training.

Usage:
    python test_metaembed_kd.py
"""

import sys
import torch
from pathlib import Path
from transformers import AutoConfig
from functools import partial

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.deep_impact.models.meta_embed_impact import MetaEmbedDeepImpact
from src.deep_impact.models.original import DeepImpact
from src.deep_impact.training.meta_embed_kd_trainer import MetaEmbedKDTrainer, meta_embed_kd_collate_fn
from src.utils.datasets import DistillationScores


def test_expansion_tokens():
    """Test 1: Expansion token initialization"""
    print("\n" + "="*60)
    print("TEST 1: Expansion Token Initialization")
    print("="*60)
    
    # Test setting expansion tokens
    MetaEmbedDeepImpact.set_expansion_tokens(64)
    
    assert len(MetaEmbedDeepImpact.expansion_tokens) == 64, "Should have 64 expansion tokens"
    assert MetaEmbedDeepImpact.expansion_tokens[0] == "exp0", "First token should be exp0"
    assert MetaEmbedDeepImpact.expansion_tokens[-1] == "exp63", "Last token should be exp63"
    
    vocab_size = MetaEmbedDeepImpact.tokenizer.get_vocab_size()
    print(f"✓ Expansion tokens set: {len(MetaEmbedDeepImpact.expansion_tokens)}")
    print(f"✓ Tokenizer vocab size: {vocab_size}")
    print(f"✓ Expected: 30522 (BERT) + 64 (expansion) = 30586")
    
    assert vocab_size == 30586, f"Expected vocab size 30586, got {vocab_size}"
    print("✓ TEST 1 PASSED")


def test_model_initialization():
    """Test 2: Model initialization"""
    print("\n" + "="*60)
    print("TEST 2: Model Initialization")
    print("="*60)
    
    # Set expansion tokens first
    MetaEmbedDeepImpact.set_expansion_tokens(64)
    
    # Create model
    config = AutoConfig.from_pretrained('Luyu/co-condenser-marco')
    model = MetaEmbedDeepImpact(
        config,
        dense_dim=128,
        num_expansion_tokens=64,
        sparse_weight=0.5,
        dense_weight=0.5,
        learn_weights=True
    )
    
    # Resize embeddings
    vocab_size = MetaEmbedDeepImpact.tokenizer.get_vocab_size()
    model.resize_token_embeddings(vocab_size)
    
    print(f"✓ Model created successfully")
    print(f"✓ Dense dimension: {model.dense_dim}")
    print(f"✓ Num expansion tokens: {model.num_expansion_tokens}")
    print(f"✓ Learn weights: {model.learn_weights}")
    print(f"✓ Embedding size: {model.bert.embeddings.word_embeddings.num_embeddings}")
    
    assert model.bert.embeddings.word_embeddings.num_embeddings == vocab_size, \
        "Embedding size should match vocab size"
    
    print("✓ TEST 2 PASSED")
    return model


def test_forward_pass(model):
    """Test 3: Forward pass"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # Forward pass without dense embeddings
    with torch.no_grad():
        sparse_scores, dense_embeddings = model(
            input_ids, 
            attention_mask, 
            token_type_ids,
            return_dense_embeddings=False
        )
    
    print(f"✓ Forward pass (sparse only):")
    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Sparse scores shape: {sparse_scores.shape}")
    print(f"  - Dense embeddings: {dense_embeddings}")
    
    assert sparse_scores.shape == (batch_size, seq_len, 1), \
        f"Expected shape ({batch_size}, {seq_len}, 1), got {sparse_scores.shape}"
    assert dense_embeddings is None, "Dense embeddings should be None when not requested"
    
    # Forward pass with dense embeddings
    with torch.no_grad():
        sparse_scores, dense_embeddings = model(
            input_ids,
            attention_mask,
            token_type_ids,
            return_dense_embeddings=True
        )
    
    print(f"✓ Forward pass (sparse + dense):")
    print(f"  - Sparse scores shape: {sparse_scores.shape}")
    print(f"  - Dense embeddings shape: {dense_embeddings.shape}")
    
    assert sparse_scores.shape == (batch_size, seq_len, 1), "Sparse scores shape mismatch"
    assert dense_embeddings.shape == (batch_size, seq_len, 128), \
        f"Expected dense shape ({batch_size}, {seq_len}, 128), got {dense_embeddings.shape}"
    
    print("✓ TEST 3 PASSED")


def test_late_interaction(model):
    """Test 4: Late interaction scoring"""
    print("\n" + "="*60)
    print("TEST 4: Late Interaction Scoring")
    print("="*60)
    
    # Create dummy embeddings
    batch_size = 2
    q_len = 32
    d_len = 64
    dense_dim = 128
    
    query_embeddings = torch.randn(batch_size, q_len, dense_dim)
    doc_embeddings = torch.randn(batch_size, d_len, dense_dim)
    query_mask = torch.ones(batch_size, q_len).bool()
    doc_mask = torch.ones(batch_size, d_len).bool()
    
    # Normalize embeddings (as done in forward)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    
    with torch.no_grad():
        scores = model.compute_late_interaction_score(
            query_embeddings,
            doc_embeddings,
            query_mask,
            doc_mask,
            temperature=model.temperature
        )
    
    print(f"✓ Late interaction scores shape: {scores.shape}")
    print(f"✓ Score values: {scores}")
    
    assert scores.shape == (batch_size,), f"Expected shape ({batch_size},), got {scores.shape}"
    assert not torch.isnan(scores).any(), "Scores contain NaN"
    assert not torch.isinf(scores).any(), "Scores contain Inf"
    
    print("✓ TEST 4 PASSED")


def test_collate_function():
    """Test 5: Collate function"""
    print("\n" + "="*60)
    print("TEST 5: Collate Function")
    print("="*60)
    
    # Create dummy batch (like DistillationScores format)
    batch = [
        ("what is python", [
            ("Python is a programming language", 9.5),
            ("Python coding tutorial", 8.2),
            ("Learn Python basics", 7.8),
            ("Java programming", 4.5),
            ("Web development", 3.2),
            ("Machine learning", 5.1),
            ("Data science", 4.8),
            ("SQL database", 3.5),
            ("JavaScript guide", 2.9),
        ]),
        ("machine learning tutorial", [
            ("Machine learning basics", 9.0),
            ("Deep learning guide", 8.5),
            ("Python ML tutorial", 7.9),
            ("Data science intro", 5.5),
            ("Statistics fundamentals", 4.2),
            ("Neural networks", 6.8),
            ("AI applications", 5.9),
            ("Computer vision", 4.5),
            ("NLP basics", 5.2),
        ])
    ]

    print("batch created")
    
    # Test collate function
    collate_fn = partial(
        meta_embed_kd_collate_fn,
        model_cls=MetaEmbedDeepImpact,
        max_length=256,
        num_hard_negatives=8,
        top_k_hard=4
    )
    print("collate function created")
    batch_dict = collate_fn(batch)

    print("batch dict created")
    
    print(f"✓ Collate function output keys: {batch_dict.keys()}")
    print(f"✓ Queries shape: {batch_dict['queries'].shape}")
    print(f"✓ Documents shape: {batch_dict['documents'].shape}")
    print(f"✓ Scores shape: {batch_dict['scores'].shape}")

    print("HEHEHEHEHE")
    
    batch_size = len(batch)
    num_docs = 1 + 8  # 1 positive + 8 negatives
    
    assert batch_dict['queries'].shape[0] == batch_size, "Batch size mismatch in queries"
    assert batch_dict['documents'].shape == (batch_size, num_docs, 256), \
        f"Expected documents shape ({batch_size}, {num_docs}, 256), got {batch_dict['documents'].shape}"
    assert batch_dict['scores'].shape == (batch_size, num_docs), "Scores shape mismatch"
    
    print(f"✓ Successfully sampled {num_docs} documents (1 pos + 8 neg)")
    print("✓ TEST 5 PASSED")
    
    return batch_dict


def test_trainer_forward(model, batch_dict):
    """Test 6: Trainer get_output_scores"""
    print("\n" + "="*60)
    print("TEST 6: Trainer Forward Pass")
    print("="*60)
    
    # Move model to CPU for testing
    model = model.cpu()
    model.eval()
    
    # Create a minimal trainer instance (without full setup)
    class MinimalTrainer:
        def __init__(self, model):
            self.model = model
            self.model.module = model  # Simulate DDP wrapping
            self.gpu_id = 'cpu'
    
    trainer = MinimalTrainer(model)
    
    # Move batch to CPU
    for key in batch_dict:
        if isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = batch_dict[key].cpu()
    
    # Test get_output_scores from MetaEmbedKDTrainer
    from src.deep_impact.training.meta_embed_kd_trainer import MetaEmbedKDTrainer
    
    with torch.no_grad():
        try:
            # Call the actual trainer method
            sparse_scores, dense_scores, combined_scores, teacher_scores = \
                MetaEmbedKDTrainer.get_output_scores(trainer, batch_dict)
            
            print(f"✓ Sparse scores shape: {sparse_scores.shape}")
            print(f"✓ Dense scores shape: {dense_scores.shape}")
            print(f"✓ Combined scores shape: {combined_scores.shape}")
            print(f"✓ Teacher scores shape: {teacher_scores.shape}")
            
            batch_size = batch_dict['queries'].shape[0]
            num_docs = batch_dict['documents'].shape[1]
            
            assert sparse_scores.shape == (batch_size, num_docs), "Sparse scores shape mismatch"
            assert dense_scores.shape == (batch_size, num_docs), "Dense scores shape mismatch"
            assert combined_scores.shape == (batch_size, num_docs), "Combined scores shape mismatch"
            assert teacher_scores.shape == (batch_size, num_docs), "Teacher scores shape mismatch"
            
            print(f"✓ Sample sparse scores: {sparse_scores[0][:3]}")
            print(f"✓ Sample dense scores: {dense_scores[0][:3]}")
            print(f"✓ Sample combined scores: {combined_scores[0][:3]}")
            
            print("✓ TEST 6 PASSED")
            return sparse_scores, dense_scores, combined_scores, teacher_scores
            
        except Exception as e:
            print(f"✗ TEST 6 FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise


def test_loss_computation(model, batch_dict, outputs):
    """Test 7: Loss computation"""
    print("\n" + "="*60)
    print("TEST 7: Loss Computation")
    print("="*60)
    
    sparse_scores, dense_scores, combined_scores, teacher_scores = outputs
    
    # Create minimal trainer for loss computation
    from src.deep_impact.training.meta_embed_kd_trainer import MetaEmbedKDTrainer
    
    class MinimalTrainer:
        def __init__(self):
            self.lambda_di = 1.0
            self.lambda_li = 1.0
            self.lambda_kd = 1.0
            self.margin = 0.2
            self.temperature = 1.0
    
    trainer = MinimalTrainer()
    
    try:
        # Test individual loss components
        loss_di = MetaEmbedKDTrainer.compute_deepimpact_loss(trainer, sparse_scores)
        print(f"✓ DeepImpact loss: {loss_di.item():.4f}")
        assert not torch.isnan(loss_di), "DeepImpact loss is NaN"
        
        loss_li = MetaEmbedKDTrainer.compute_late_interaction_loss(trainer, dense_scores)
        print(f"✓ Late interaction loss: {loss_li.item():.4f}")
        assert not torch.isnan(loss_li), "Late interaction loss is NaN"
        
        loss_kd = MetaEmbedKDTrainer.compute_kd_loss(
            trainer, combined_scores, sparse_scores, dense_scores, teacher_scores
        )
        print(f"✓ Knowledge distillation loss: {loss_kd.item():.4f}")
        assert not torch.isnan(loss_kd), "KD loss is NaN"
        
        # Compute total loss
        total_loss = trainer.lambda_di * loss_di + trainer.lambda_li * loss_li + trainer.lambda_kd * loss_kd
        print(f"✓ Total loss: {total_loss.item():.4f}")
        assert not torch.isnan(total_loss), "Total loss is NaN"
        
        print("✓ TEST 7 PASSED")
        
    except Exception as e:
        print(f"✗ TEST 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("METAEMBED-KD SMOKE TEST SUITE")
    print("="*60)
    print("Testing model initialization, forward pass, and training components")
    print("This ensures everything works before launching full training.")
    
    try:
        # Run tests in sequence
        test_expansion_tokens()
        model = test_model_initialization()
        test_forward_pass(model)
        test_late_interaction(model)
        batch_dict = test_collate_function()
        outputs = test_trainer_forward(model, batch_dict)
        test_loss_computation(model, batch_dict, outputs)
        
        # Summary
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED! 🎉")
        print("="*60)
        print("Your MetaEmbed-KD setup is working correctly.")
        print("You can now run full training with confidence!")
        print("\nTo run training:")
        print("  sbatch slurm_jobs/training_metaembed_kd.sh")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        print("\nPlease fix the issue before running full training.")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_all_tests()
    sys.exit(0 if success else 1)


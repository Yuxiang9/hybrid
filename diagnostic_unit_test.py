import pytest
import torch
import sys
sys.path.append("/scratch/yx3044/Projects/improving-learned-index")

from src.deep_impact.models.meta_embed_impact import MetaEmbedDeepImpact
from src.deep_impact.training.meta_embed_kd_trainer import meta_embed_kd_collate_fn
from transformers import AutoConfig

# ----------------------
# Helper for printing diagnostics clearly
# ----------------------
def print_diag(title, info):
    print("\n" + "="*80)
    print(f"DIAG: {title}")
    print("-"*80)
    if isinstance(info, dict):
        for k,v in info.items():
            print(f"{k}: {v}")
    else:
        print(info)
    print("="*80 + "\n")


# -------------------------------------------------------------------------
# FIXED SETUP FOR DIAGNOSTIC PURPOSES
# -------------------------------------------------------------------------

@pytest.fixture
def setup_debug_model():
    """
    Creates a MetaEmbed model with expansion tokens and returns
    (model, tokenizer, num_exp, max_length).
    """
    num_exp = 8
    max_length = 32
    dense_dim = 16

    # Expand tokenizer BEFORE creating model
    MetaEmbedDeepImpact.set_expansion_tokens(num_exp)

    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = MetaEmbedDeepImpact(config, dense_dim=dense_dim, num_expansion_tokens=num_exp)

    # Resize embeddings AFTER tokenizer expansion
    model.resize_token_embeddings(MetaEmbedDeepImpact.tokenizer.vocab_size)

    return model, MetaEmbedDeepImpact.tokenizer, num_exp, max_length


# -------------------------------------------------------------------------
# TEST 1 — Check tokenizer mismatch between collate_fn and model
# -------------------------------------------------------------------------

def test_tokenizer_consistency(setup_debug_model):
    model, tokenizer, num_exp, max_length = setup_debug_model

    # Get model-side tokenizer (tokenizer tied to BERT embeddings)
    model_tok = tokenizer              # class-level tokenizer
    collate_tok = MetaEmbedDeepImpact.tokenizer   # collate_fn uses this one

    diag = {
        "model_vocab": model_tok.vocab_size,
        "collate_vocab": collate_tok.vocab_size,
        "tokenizer_objects_same": (model_tok is collate_tok),
        "exp_tokens": MetaEmbedDeepImpact.expansion_tokens,
        "exp_ids_model": [model_tok.convert_tokens_to_ids(t) for t in MetaEmbedDeepImpact.expansion_tokens],
        "exp_ids_collate": [collate_tok.convert_tokens_to_ids(t) for t in MetaEmbedDeepImpact.expansion_tokens],
    }

    print_diag("TOKENIZER CONSISTENCY", diag)

    # We don't assert failure — we just report mismatch.
    assert True


# -------------------------------------------------------------------------
# TEST 2 — Run collate_fn and detect OUT-OF-RANGE token ids
# -------------------------------------------------------------------------

def test_detect_token_id_overflow(setup_debug_model):
    model, tokenizer, num_exp, max_length = setup_debug_model

    batch = [
        (
            "what is ai",
            [
                ("Artificial intelligence helps machines learn.", 5.0),
                ("Birds migrate seasonally.", 0.5)
            ]
        )
    ]

    out = meta_embed_kd_collate_fn(
        batch=batch,
        model_cls=MetaEmbedDeepImpact,
        max_length=max_length,
        num_hard_negatives=1,
        top_k_hard=1,
    )

    doc_ids = out["documents"]
    max_id = doc_ids.max().item()
    emb_rows = model.bert.embeddings.word_embeddings.num_embeddings

    diag = {
        "max_token_id": max_id,
        "embedding_rows": emb_rows,
        "overflow": max_id >= emb_rows,
        "exp_ids": [tokenizer.convert_tokens_to_ids(t) for t in MetaEmbedDeepImpact.expansion_tokens],
        "sample_doc_ids_tail": doc_ids[0,0,-num_exp:].tolist(),
    }

    print_diag("TOKEN ID OVERFLOW CHECK", diag)

    # Always true; this test never fails — only prints diagnostics
    assert True


# -------------------------------------------------------------------------
# TEST 3 — Check if checkpoint loading breaks embedding size
# -------------------------------------------------------------------------

def test_embedding_resize_after_checkpoint(setup_debug_model):
    model, tokenizer, num_exp, max_length = setup_debug_model

    initial_size = model.bert.embeddings.word_embeddings.num_embeddings

    # Fake checkpoint load (simulate your code path)
    # Replace with ModelCheckpoint.load if necessary
    # Here we simulate “wrong” load restoring old embeddings:
    fake_old_embedding = torch.nn.Embedding(30522, model.bert.embeddings.word_embeddings.embedding_dim)
    model.bert.embeddings.word_embeddings = fake_old_embedding

    post_load_size = model.bert.embeddings.word_embeddings.num_embeddings

    diag = {
        "initial_emb_size": initial_size,
        "post_load_emb_size": post_load_size,
        "shrunk_after_checkpoint": post_load_size < initial_size
    }

    print_diag("CHECKPOINT EMBEDDING SIZE CHECK", diag)

    assert True


if __name__ == "__main__":
    pytest.main([__file__])
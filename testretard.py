import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.deep_impact.models.meta_embed_impact import MetaEmbedDeepImpact
from src.deep_impact.training.meta_embed_kd_trainer import (
    MetaEmbedKDTrainer,
    meta_embed_kd_collate_fn
)

from transformers import AutoConfig


class DummyKDDataset(Dataset):
    """
    Each sample is:
        (
            "dummy query",
            [("doc0 text", score0), ("doc1 text", score1), ...]
        )
    """
    def __init__(self, num_samples=2, num_docs=3):
        self.num_samples = num_samples
        self.num_docs = num_docs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        query = f"dummy query {idx}"
        docs = []
        for d in range(self.num_docs):
            text = f"document number {d} for idx {idx}"
            score = float(self.num_docs - d)  # decreasing teacher scores
            docs.append((text, score))
        return query, docs


print("Initializing MetaEmbedDeepImpact...")

config = AutoConfig.from_pretrained("Luyu/co-condenser-marco")

# add expansion token
MetaEmbedDeepImpact.configure_expansion_token(8)

# create model
model = MetaEmbedDeepImpact(
    config,
    dense_dim=16,
    num_expansion_tokens=8,
)

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

exp_id = MetaEmbedDeepImpact.exp_id
print(f"Expansion token ID: {exp_id}")
print(f"Tokenizer vocab size: {len(MetaEmbedDeepImpact.tokenizer)}")


print("Running forward pass smoke test...")

vocab_size = len(MetaEmbedDeepImpact.tokenizer)

ids = torch.randint(0, vocab_size, (1, 16)).to(device)
mask = torch.ones_like(ids).to(device)
types = torch.zeros_like(ids).to(device)

sparse, dense = model(ids, mask, types, return_dense_embeddings=True)

print("Sparse shape:", sparse.shape)   # [1, 16, 1]
print("Dense shape: ", dense.shape)    # [1, 16, 16]

assert dense.abs().sum() > 0.001, "Dense embeddings appear zero!"
print("✓ Dense embeddings non-zero")


dataset = DummyKDDataset(num_samples=2, num_docs=3)

collate = lambda batch: meta_embed_kd_collate_fn(
    batch,
    model_cls=MetaEmbedDeepImpact,
    max_length=32,
    num_hard_negatives=2,
    top_k_hard=1
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate,
)
batch = next(iter(loader))

print("Batch keys:", batch.keys())
print("Query IDs shape:", batch["queries"].shape)
print("Doc IDs shape:", batch["documents"].shape)



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = MetaEmbedKDTrainer(
    model=model,
    optimizer=optimizer,
    train_data=loader,
    checkpoint_dir=".",
    batch_size=2,
    save_every=99999,
    save_best=False,
    gradient_accumulation_steps=1,
    lambda_di=1.0,
    lambda_li=1.0,
    lambda_kd=1.0,
)

# simulate DDP-style API (trainer assumes model.module exists)
trainer.model = nn.DataParallel(model)


print("Running trainer forward pass...")

sparse, dense, combined, teacher = trainer.get_output_scores(batch)

print("Sparse scores:\n", sparse)
print("Dense scores:\n", dense)
print("Combined scores:\n", combined)
print("Teacher scores:\n", teacher)

assert dense.abs().sum() > 0.001, "Dense scores appear zero!"
assert not torch.allclose(teacher, combined), "Teacher equals student! KD broken!"

print("✓ Trainer forward pass OK")



print("Running backward pass...")

loss = trainer.evaluate_loss((sparse, dense, combined, teacher), batch)
print("Loss:", loss.item())

loss.backward()

# check gradients exist
num_params = sum(p.numel() for p in model.parameters())
num_grads = sum(p.grad is not None for p in model.parameters())

print(f"Params with gradient: {num_grads}/{num_params}")

assert num_grads > 0, "No gradients flowed!"

optimizer.step()
optimizer.zero_grad()

print("✓ Backprop and optimizer step successful")


print("Checking weight updates...")

with torch.no_grad():
    ws_before = model.log_sparse_weight.clone()
    wd_before = model.log_dense_weight.clone()

# run one more training step
sparse, dense, combined, teacher = trainer.get_output_scores(batch)
loss = trainer.evaluate_loss((sparse, dense, combined, teacher), batch)
loss.backward()
optimizer.step()

with torch.no_grad():
    ws_after = model.log_sparse_weight
    wd_after = model.log_dense_weight

print("Sparse weight before/after:", ws_before.item(), ws_after.item())
print("Dense weight before/after:", wd_before.item(), wd_after.item())

assert ws_before.item() != ws_after.item(), "Sparse weight did not update!"
assert wd_before.item() != wd_after.item(), "Dense weight did not update!"

print("✓ Weight update verified")
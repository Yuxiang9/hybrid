import torch
from src.deep_impact.models import DeepImpact
from src.deep_impact.training.trainer import Trainer

model = DeepImpact.load()
model.train()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

triples_path = "/scratch/yx3044/Projects/improving-learned-index/required_files/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl"
queries_path = "/scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/queries.train.exp32.tsv"
collection_path = "/scratch/yx3044/Projects/improving-learned-index/expansions/collection.exp32.tsv"

from src.deep_impact.train import collate_fn, MSMarcoTriples
triples = MSMarcoTriples(triples_path, queries_path, collection_path)
batch = [triples[0]]
batch = collate_fn(batch, max_length=300)

inp_ids, attn, type_ids = Trainer.get_input_tensors(None, batch['encoded_list'])

out = model(inp_ids, attn, type_ids)                    # shape (1 , L , 1)
mask = batch['masks']
score = (mask * out).sum()                              # scalar loss surrogate
score.backward()

with torch.no_grad():
    ids = [DeepImpact.tokenizer.token_to_id(f"exp{i}") for i in range(32)]
    grads = model.bert.embeddings.word_embeddings.weight.grad[ids]  # (32 , 768)

print("grad L2 norm per token:", grads.norm(dim=1))



def impact_vector():
    with torch.no_grad():
        eids = torch.tensor(ids).to(model.device)
        emb  = model.bert.embeddings.word_embeddings(eids)         # (32 , 768)
        return model.impact_score_encoder(emb).cpu().squeeze(-1)   # (32,)

print("initial scores:", impact_vector())
for _ in range(100):           # ~1 second on GPU
    optim.zero_grad()
    score = (mask * model(inp_ids, attn, type_ids)).sum()
    (-score).backward()        # dummy objective
    optim.step()
print("after 100 steps:", impact_vector())



# from src.deep_impact.models import DeepImpact
# tok = DeepImpact.tokenizer
# print(tok.encode("<EXP5>").tokens)             # should be ['exp5']
# print(tok.token_to_id("exp5"))               # should be a valid id, not None


# import torch

# # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# # batch = next(iter(train_loader))
# # ids, _, _ = Trainer.get_input_tensors(None, batch["encoded_list"])
# # mask = batch["masks"][0]          # first doc, shape (L,1)

# # # Which positions are 1?
# # mask_pos = torch.where(mask.squeeze()==1)[0].tolist()

# # # Map positions to tokens
# # tokens = [DeepImpact.tokenizer.id_to_token(int(i)) for i in ids[0]]
# # print([(p, tokens[p]) for p in mask_pos])

# CHECKPOINT_PATH = "/scratch/yx3044/Projects/improving-learned-index/checkpoints6.1/DeepImpact_2000.pt"
# model = DeepImpact.load(CHECKPOINT_PATH)


# with torch.no_grad():
#     exp_vec = model.bert.embeddings.word_embeddings.weight[DeepImpact.tokenizer.token_to_id("<EXP5>")]
#     pre_relu = model.impact_score_encoder[0](exp_vec)
#     print("before ReLU", pre_relu)
#     print("after ReLU", torch.relu(pre_relu))


import numpy as np
from matplotlib import pyplot as plt
range_alpha = np.arange(0, 3, 0.2)
# rsa_metrics_list = [0.53419, 0.62108, 0.63633, 0.65096, 0.66933, 0.67263, 0.67205, 0.66687, 0.66571, 0.65462, 0.65429, 0.63205, 0.62094, 0.60872, 0.59494]

rsa_metrics_list = [0.42, 0.51333, 0.53, 0.54, 0.55333, 0.54667, 0.54667, 0.53, 0.52333, 0.50667, 0.51, 0.48333, 0.47, 0.45, 0.43]

plt.plot(range_alpha, rsa_metrics_list)
plt.axhline(y=0.54667, color='r', linestyle='--', label='BM25')
plt.text(-0.2, 0.54667, '0.54667', verticalalignment='center')
# plt.axhline(y=0.67401, color='r', linestyle='--', label='BM25')
# plt.text(-0.2, 0.67401, '0.67401', verticalalignment='center')

plt.legend()
for i, (x, y) in enumerate(zip(range_alpha, rsa_metrics_list)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel("alpha")
plt.ylabel("NDCG@1")
plt.title("RSA re-weighted with different alpha")
plt.savefig("rsa_alpha_map_2.png")




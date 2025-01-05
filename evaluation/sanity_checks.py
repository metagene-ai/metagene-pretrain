from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from litgpt import LLM
from litgpt.utils import chunked_cross_entropy

# TO USE THIS SCRIPT, PLEASE CHANGE THE FOLLOWING DIRECTORIES
DATASET_DIR = "data/sample_reads.csv"
CKPT_DIR = "PATH/TO/CKPT"

CTX_LEN = 12  # Context Length
GEN_LEN = 20  # Generation Length

print("START: Loading Dataset")
dataset = [line.strip() for line in open(DATASET_DIR, "r")]
print("SUCCESS: Dataset Loaded:")
for seq in dataset[:5]:
    print("\t", seq)

B = len(dataset)
tokenizer = AutoTokenizer.from_pretrained("metagene-ai/METAGENE-1")
llm = LLM.load(CKPT_DIR)

print('\n\n\nSTART: "Validation" loss calculation')
input_ids = tokenizer(
    dataset, return_tensors="pt", padding=True, add_special_tokens=False
).input_ids
input_ids = input_ids.to(llm.model.device)
logits = llm.model(input_ids)
target_ids = input_ids.clone()
target_ids[target_ids == tokenizer.pad_token_id] = -100
loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
print("SUCCESS: Validation Loss:", loss.item())

# Test Generation
print("\n\n\nSTART: Generation+Ranking Test")
all_ranks = []

for sample_idx in range(0, B):
    ctx = input_ids[sample_idx : sample_idx + 1, :CTX_LEN].to(llm.model.device)
    ranks = []
    for tok_idx in tqdm(range(min(input_ids.shape[1] - CTX_LEN, GEN_LEN))):
        if input_ids[sample_idx, CTX_LEN + tok_idx] == 0:
            break
        logits = llm.model(ctx)[:, -1, :]
        # Evaluate rank of the ground truth token to the logits
        ground_truth_token_id = input_ids[sample_idx, CTX_LEN + tok_idx].item()
        rank = (
            torch.argsort(logits, descending=True)
            .squeeze()
            .tolist()
            .index(ground_truth_token_id)
        )
        if rank > 500:
            print("Ground Truth Token:", tokenizer.decode([ground_truth_token_id]))
        ranks.append(rank)
        # Use ground truth token to update context
        ctx = torch.cat(
            [
                ctx,
                input_ids[sample_idx : sample_idx + 1, CTX_LEN + tok_idx]
                .unsqueeze(1)
                .to(llm.model.device),
            ],
            dim=1,
        )
    print(ranks)
    all_ranks.append(ranks)

print("\n\n\nSAVING ranks to all_ranks.txt")
# save all ranks, each row is a rank of current sequence
with open("all_ranks.txt", "w") as f:
    for ranks in all_ranks:
        f.write(",".join(map(str, ranks)) + "\n")

rank_avgs = [sum(ranks) / len(ranks) for ranks in all_ranks]
avg_rank = sum(rank_avgs) / len(rank_avgs)
print("Average Rank:", round(avg_rank, 2))

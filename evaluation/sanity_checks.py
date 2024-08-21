# Load Dataset
import random
from collections import Counter
from tqdm import tqdm

import torch
from transformers import PreTrainedTokenizerFast
from litgpt import LLM
from litgpt.utils import chunked_cross_entropy

# TO USE THIS SCRIPT, PLEASE CHANGE THE FOLLOWING DIRECTORIES
DATASET_DIR = "../../mgfm_sanity_check/dataset/cleaned_tokens_2000000000.txt"
CKPT_DIR = "/Users/oliverliu/mgfm_ckpt"

N = 1000  # Dataset Size
B = 32  # Batch Size
CTX_LEN = 12  # Context Length
GEN_LEN = 20  # Generation Length

print("START: Loading Dataset")
random.seed(42)

dataset = []
with open(DATASET_DIR, "r") as f:
    i = 0
    for line in f:
        dataset.append("_" + line.strip())
        i += 1
        if i == 100000:
            break
print("SUCCESS: Dataset Loaded:")
random.shuffle(dataset)
dataset = dataset[:N]
for seq in dataset[:5]:
    print(seq)

print("START: Tokenizer Consistency Check")
tokenizer_hf = PreTrainedTokenizerFast.from_pretrained(CKPT_DIR)
tokenizer_hf.pad_token = "[PAD]"
tokenizer_hf.pad_token_id = 0

# Load lit model
llm = LLM.load(CKPT_DIR)
start_token_ids = []
for i in range(len(dataset)):
    litgpt_token_ids = llm.preprocessor.encode(dataset[i]).tolist()
    hf_token_ids = (
        tokenizer_hf.encode(dataset[i], return_tensors="pt").squeeze().tolist()
    )
    assert (
        litgpt_token_ids == hf_token_ids
    ), f"ERROR: Tokenizers are different at index {i}"
    start_token_ids.append(litgpt_token_ids[1])
print("SUCCESS: Tokenizers are the same")
most_common_start_token_ids = Counter(start_token_ids).most_common(5)
print("Most Common Start Token IDs:")
for token_id, count in most_common_start_token_ids:
    print(tokenizer_hf.decode([token_id]), count)

print('\n\n\nSTART: "Validation" loss calculation')
batch = dataset[:B]
input_ids = tokenizer_hf(batch, return_tensors="pt", padding=True).input_ids[:, 1:]
input_ids = input_ids.to(llm.model.device)
logits = llm.model(input_ids)
target_ids = input_ids.clone()
# set -100 to padding tokens
target_ids[target_ids == 0] = -100
loss = chunked_cross_entropy(logits[..., :-1, :], target_ids[..., 1:])
print("Validation Loss:", loss.item())

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
            print("Ground Truth Token:", tokenizer_hf.decode([ground_truth_token_id]))
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
print("Average Rank:", avg_rank)
print("Rank Averages:", rank_avgs)

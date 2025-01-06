from argparse import ArgumentParser
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from litgpt import LLM
from litgpt.utils import chunked_cross_entropy


def main(args):
    print("START: Loading Dataset")
    dataset = [line.strip() for line in open(args.dataset_dir, "r")]
    print("SUCCESS: Dataset Loaded:")
    for seq in dataset[:5]:
        print("\t", seq)

    B = len(dataset)
    tokenizer = AutoTokenizer.from_pretrained("metagene-ai/METAGENE-1")
    llm = LLM.load(args.ckpt_dir)

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
        ctx = input_ids[sample_idx : sample_idx + 1, : args.ctx_len].to(
            llm.model.device
        )
        ranks = []
        for tok_idx in tqdm(
            range(min(input_ids.shape[1] - args.ctx_len, args.gen_len))
        ):
            if input_ids[sample_idx, args.ctx_len + tok_idx] == 0:
                break
            logits = llm.model(ctx)[:, -1, :]
            # Evaluate rank of the ground truth token to the logits
            ground_truth_token_id = input_ids[sample_idx, args.ctx_len + tok_idx].item()
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
                    input_ids[sample_idx : sample_idx + 1, args.ctx_len + tok_idx]
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/sample_reads.csv")
    parser.add_argument("--ckpt_dir", type=str, default="PATH/TO/CKPT")
    parser.add_argument("--ctx_len", type=int, default=12)
    parser.add_argument("--gen_len", type=int, default=20)
    args = parser.parse_args()
    main(args)

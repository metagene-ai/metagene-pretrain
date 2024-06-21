# %%
import os
import gzip
import time
from random import shuffle

PREFIX = f"{os.environ['HOME']}/temp"
BUDGET = 2_000_000_000
FILES = ["jr.csv", "mj.csv"]


def prepare_state_dict():
    stat_dict = {}
    for file in FILES:
        with open(f"metadata/{file}", "r") as f:
            for line in f:
                if line.strip().endswith("cleaned/"):
                    prefix = line.strip().split()[-1]
                if line.strip().endswith("collapsed.gz"):
                    line_split = line.strip().split()
                    stat_dict[f"{prefix}{line_split[-1]}"] = int(line_split[-2])

    total_file_size = sum(stat_dict.values())
    for key in stat_dict:
        stat_dict[key] = (
            stat_dict[key],  # file size (in bytes)
            int(
                stat_dict[key] / total_file_size * BUDGET
            ),  # number of tokens to be sampled
        )
    return stat_dict


def gather_token_count_by_line(fname):
    counts = []
    with gzip.open(fname, "rt") as f:
        i = 0
        for line in f:
            if line[0] not in ["A", "C", "G", "T"]:
                continue
            counts.append((i, len(line.strip())))
            i += 1
    return counts


def determine_sample_size(counts, budget):
    total_so_far = 0
    for i in range(len(counts)):
        total_so_far += counts[i][1]
        if total_so_far > budget:
            return i


def sample_tokens(fname, counts, fout):
    sampled_index, _ = counts.pop(0)
    with gzip.open(fname, "rt") as f:
        i = 0
        for line in f:
            if line[0] not in ["A", "C", "G", "T"]:
                continue
            if i == sampled_index:
                fout.write(line)
                if counts:
                    sampled_index, _ = counts.pop(0)
                else:
                    break
            i += 1


def main(out_folder="data"):
    out_progress = f"{out_folder}/progress_{BUDGET}.txt"
    out_fname = f"{out_folder}/sampled_tokens_{BUDGET}.txt"
    stat_dict = prepare_state_dict()

    current_progress = []
    if os.path.exists(out_progress):
        with open(out_progress, "r") as f:
            for line in f:
                current_progress.append(line.split(',')[0].strip())

    fout = open(out_fname, "a")
    for key, (_, file_budget) in sorted(stat_dict.items(), key=lambda x: x[1][0]):
        if key in current_progress:
            print('Skipping', key)
            continue
        local_path = f"{PREFIX}/{os.path.basename(key)}"
        if os.path.exists(local_path):
            pass
        else:
            os.system(f"aws s3 cp s3://{key} {PREFIX}")
        counts = gather_token_count_by_line(f"{PREFIX}/{os.path.basename(key)}")
        shuffle(counts)

        sample_size = determine_sample_size(counts, file_budget)

        counts = counts[:sample_size]
        counts.sort()

        sample_tokens(f"{PREFIX}/{os.path.basename(key)}", counts, fout)
        fout.flush()

        with open(out_progress, "a") as f:
            f.write(f"{key}, {sample_size}\n")
        os.remove(local_path)


if __name__ == "__main__":
    main()

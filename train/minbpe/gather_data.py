import os, time
import numpy as np
from tqdm import tqdm
from random import shuffle


PREFIX = "/home/oliver/temp/"
BUDGET = 150_000_000

def main(args):
    files = ["jr.csv", "mj.csv"]
    stat_dict = {"jr": {}, "mj": {}}

    for file in files:
        source = file.split(".")[0]
        prefix = None
        with open(f'metadata/{file}', "r") as f:
            for line in f:
                if line.strip().endswith("cleaned/"):
                    prefix = line.strip().split()[-1]
                if line.strip().endswith("collapsed.gz"):
                    line_split = line.strip().split()
                    stat_dict[source][f"{prefix}{line_split[-1]}"] = int(line_split[-2])

    total = 0
    _stat_dict = {}
    for source, source_dict in stat_dict.items():
        for key, value in source_dict.items():
            if value <= args.limit:
                total += value
                _stat_dict[key] = value
    # total = sum(sum(stat_dict[source].values()) for source in ["jr", "mj"])

    # for source, source_dict in stat_dict.items():
    #     for key, value in source_dict.items():
    #         source_dict[key] = [value, int(BUDGET * value / total)]

    stat_dict = _stat_dict
    for key, value in stat_dict.items():
        stat_dict[key] = [value, int(BUDGET * value / total)]
    
    print('Total size:', total, 'Total files: ', len(stat_dict))

    # pbar = tqdm(total=len(stat_dict["jr"]) + len(stat_dict["mj"]))
    pbar = tqdm(total=len(stat_dict))
    done_runs = []
    # create if not exists
    if not os.path.exists(f"{PREFIX}done_runs.txt"):
        with open(f"{PREFIX}done_runs.txt", "w") as f:
            pass
    
    with open(f"{PREFIX}done_runs.txt", "r") as f:
        for line in f:
            done_runs.append(line.strip())

    for key in sorted(stat_dict.keys()):
        value = stat_dict[key]
        pbar.set_description(f"Processing {key}; {value[1]} tokens")
        fname = key.split("/")[-1].split(".gz")[0]
        if fname in done_runs:
            pbar.update(1)
            print(f"Skipping {fname}")
            continue
        if not os.path.exists(f"{PREFIX}{fname}-shuffled"):
            # use aws s3 cp to copy files
            start = time.time()
            if not os.path.exists(f"{PREFIX}{fname}.gz"):
                os.system(f"aws s3 cp s3://{key} {PREFIX}")
            print(f"Downloading {fname} took {time.time() - start:.2f} seconds")
            # decompress .gz files
            start = time.time()
            if not os.path.exists(f"{PREFIX}{fname}"):
                os.system(f"gunzip {PREFIX}{fname}.gz")
            print(f"Decompressing {fname} took {time.time() - start:.2f} seconds")

            start = time.time()
            if not os.path.exists(f"{PREFIX}{fname}-cleaned"):
                lines = []
                with open(f"{PREFIX}{fname}", "r") as fin, open(f"{PREFIX}{fname}-cleaned", "w") as fout:
                    ct = 0
                    line = fin.readline()
                    while line:
                        if line[0] in ["A", "C", "G", "T"]:
                            ct += 1
                            fout.write(line)
                            lines.append(line)
                        line = fin.readline()
                
            else:
                lines = open(f"{PREFIX}{fname}-cleaned", "r").readlines()
            print(f"Cleaning {fname} took {time.time() - start:.2f} seconds")
            os.system(f"rm {PREFIX}{fname}")  
            # shuffle file
            start = time.time()
            shuffle(lines)
            print(f"Shuffling {fname} took {time.time() - start:.2f} seconds")
            os.system(f"rm {PREFIX}{fname}-cleaned")
            with open(f"{PREFIX}{fname}-shuffled", "w") as f:
                f.writelines(lines)
        else:
            print(f"Skipping {fname}-shuffled")
        
        # sample value[1] tokens
        fout = open(f"{PREFIX}final.txt", "a")
        subpbar = tqdm(total=value[1])
        with open(f"{PREFIX}{fname}-shuffled", "r") as fin, open(f"{PREFIX}final.txt", "a") as fout:
            line = fin.readline()
            added_tokens = 0
            while line:
                if added_tokens + len(line.strip()) > value[1]:
                    break
                fout.write(line)
                added_tokens += len(line.strip())
                subpbar.update(len(line.strip()))                
                line = fin.readline()
        with open(f"{PREFIX}done_runs.txt", "a") as f:
            f.write(f"{fname}\n")
        os.system(f"rm {PREFIX}{fname}-shuffled")
        pbar.update(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5e9)
    args = parser.parse_args()
    main(args)
from typing import List
from streaming import MDSWriter
import torch
from litgpt.utils import CLI
from pathlib import Path
from torch.utils.data import random_split



def get_data(deduplication: bool, download_dir: Path) -> None:

        def parse_human_virus_ids(fname: Path) -> List[str]:
            shard = []
            with open(fname, "r") as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if not line[1].startswith("M_"):
                        continue
                    shard.append(line[1])
            return shard

        def parse_seq_reads(fname: Path) -> List[str]:
            shard = []
            human_virus_shard = []
            skip_read = False
            with open(fname, "r") as f:
                for line in f.readlines():
                    if line[0] in ["A", "C", "G", "T"]:
                        if skip_read:
                            skip_read = False
                            human_virus_shard.append(line.strip())
                        else:
                            shard.append(line.strip())
            return shard, human_virus_shard

        data = []
        # for fname in self.download_dir.glob("*-cleaned-*.collapsed"):
        for fname in download_dir.glob("*.txt"):
            shard, _ = parse_seq_reads(fname)
            data += shard

        if deduplication:
            original_len = len(data)
            data = list(set(data))
            print(f"Removed {original_len - len(data)} duplicates.")

        return data


def main(
    data_dir = Path("data/nao_mosaic"), # Local or remote directory in which to store the compressed output files
    original_data= Path("data/nao"),
    deduplication: bool = True,
    val_split_fraction: float = 0.02,
    seed: int = 42,
):
    # A dictionary mapping input fields to their data types
    columns = {'text': 'str'}

    # Shard compression, if any
    compression = 'zstd'


    data = get_data(deduplication, original_data)
        # Partition the dataset into train and test
    train_data, test_data = random_split(
        data,
        [1.0 - val_split_fraction, val_split_fraction],
        generator=torch.Generator().manual_seed(seed),
    )
    train_data, test_data = list(train_data), list(test_data)
    # Save the samples as shards using MDSWriter

    for data, name in [(train_data, "train"), (test_data, "test")]:
        with MDSWriter(out=str(data_dir / name), columns=columns, compression=compression) as out:
            for line in data:
                sample = {'text': line}
                out.write(sample)

    print("done")

if __name__ == "__main__":
    CLI(main)


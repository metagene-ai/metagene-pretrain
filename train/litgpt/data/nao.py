# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List

import torch
from torch.utils.data import DataLoader, random_split

from litgpt import Tokenizer
# TODO: potentially implement MLMDataset
from litgpt.data import DataModule, NAODataset, get_sft_collate_fn

# Our current implementation roughly follows the Alpaca data module
# TODO: implement s3 streaming dataset for NAO
@dataclass
class NAO(DataModule):
    """The TinyLlama data module is composed of a mix of SlimPajama and Starcoder data.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.02
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/nao")
    """The directory in which the downloaded dataset gets saved."""

    # data_path: Union[str, Path] = Path("data/")
    # """The path to the data directory, containing two folders 'slimpajama' and 'starcoder'
    # which are the output of the preprocessing step done in advance. See the `tutorial/pretrain_tinyllama.md`
    # for instructions. The path can also be a remote path (e.g., s3://)."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[NAODataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[NAODataset] = field(default=None, init=False, repr=False)
    deduplication: bool = True
    collect_human_virus: bool = True

    def __post_init__(self):
        # Could be a remote path (s3://) or a local path
        # TODO: replace with NAO data paths
        # self.slimpajama_train = str(self.data_path).rstrip("/") + "/slimpajama/train"
        # self.slimpajama_val = str(self.data_path).rstrip("/") + "/slimpajama/val"
        # self.starcoder_train = str(self.data_path).rstrip("/") + "/starcoder"
        return

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        # for path in (self.slimpajama_train, self.slimpajama_val, self.starcoder_train):
        #     if not path.startswith("s3://") and not Path(path).is_dir():
        #         raise FileNotFoundError(
        #             "The data path for TinyLlama is expected to be the directory containing these subdirectories:"
        #             f" `slimpajama/train`, `slimpajama/val`, `starcoder`. The directory {path} does not exist."
        #             " Set it via `--data.data_path=...`"
        #         )
        return
    
    def setup(self) -> None:

        def parse_human_virus_ids(fname: Path) -> List[str]:
            shard = []
            with open(fname, "r") as f:
                for line in f.readlines():
                    line = line.strip().split()
                    if not line[1].startswith("M_"):
                        continue
                    shard.append(line[1])
            return shard

        def parse_seq_reads(fname: Path, human_virus_ids: List[str]) -> List[str]:
            shard = []
            human_virus_shard = []
            skip_read = False
            with open(fname, "r") as f:
                for line in f.readlines():
                    if line.startswith("@"):
                        id = line.strip().split()[0][1:]
                        if id in human_virus_ids:
                            skip_read = True
                            continue
                    if line[0] in ["A", "C", "G", "T"]:
                        if skip_read:
                            skip_read = False
                            human_virus_shard.append(line.strip())
                        else:
                            shard.append(line.strip())
            return shard, human_virus_shard
        
        human_virus_ids = []
        if self.collect_human_virus:
            for fname in self.download_dir.glob("*-allmatches-*.allmatches.tsv"):
                human_virus_ids += parse_human_virus_ids(fname)

        data, human_virus_data = [], []
        for fname in self.download_dir.glob("*-cleaned-*.collapsed"):
            shard, human_virus_shard = parse_seq_reads(fname, human_virus_ids)
            data += shard
            human_virus_data += human_virus_shard

        if self.deduplication:
            original_len = len(data)
            data = list(set(data))
            print(f"Removed {original_len - len(data)} duplicates.")

        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = NAODataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = NAODataset(
            data=test_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            ignore_index=self.ignore_index,
        )

        if self.collect_human_virus:
            self.human_virus_dataset = NAODataset(
                data=human_virus_data,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            )
        else:
            self.human_virus_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, 
                ignore_index=self.ignore_index, 
                pad_id=self.tokenizer.processor.pad_token_id,
            ),
        )

    def val_dataloader(self) -> List[DataLoader]:
        dataloaders = [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=get_sft_collate_fn(
                    max_seq_length=self.max_seq_length, 
                    ignore_index=self.ignore_index,
                    pad_id=self.tokenizer.processor.pad_token_id,
                ),
            )
        ]
        if self.human_virus_dataset is not None:
            dataloaders.append(
                DataLoader(
                    self.human_virus_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=get_sft_collate_fn(
                        max_seq_length=self.max_seq_length, 
                        ignore_index=self.ignore_index,
                        pad_id=self.tokenizer.processor.pad_token_id,
                    ),
                )
            
            )
        return dataloaders



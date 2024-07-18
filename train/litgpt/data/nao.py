# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
import numpy as np
import ast
from streaming import StreamingDataset, StreamingDataLoader
import streaming
from streaming.base.stream import Stream

import torch
from torch.utils.data import Dataset, DataLoader

from litgpt import Tokenizer
# TODO: potentially implement MLMDataset

from litgpt.data import DataModule, get_sft_collate_fn



class NAODataset(StreamingDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int,
        *,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        max_seq_length: int = -1,
        ignore_index: int = -100,
        split: Optional[str] = None,
        streaming_kwargs: Dict[str, Any] = {},
        context_stuffing: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(batch_size=batch_size, streams=streams, remote=remote, local=local, split=split, **streaming_kwargs)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        self.context_stuffing = context_stuffing
        self.rng = np.random.RandomState(seed=seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = super().__getitem__(idx)["token_ids"]
        toks = torch.tensor(ast.literal_eval(example))
        toks = toks[:self.max_seq_length]
        if not self.context_stuffing:
            labels = toks.clone()
            return {"input_ids": toks.type(torch.int64), "labels": labels.type(torch.int64)}
        else:
            seqlens = []
            remaining_toks_cnt = self.max_seq_length - len(toks)
            seqlens.append(len(toks))
            idx_mult = 0
            while remaining_toks_cnt > 0:
                idx_mult += 1
                new_entry = super().__getitem__(idx_mult * self.batch_size + idx)["token_ids"]
                #todo(sami): discuss is batch_size + idx is reasonable or it it should be more random
                #todo(willie): one proposal above
                new_toks = torch.tensor(ast.literal_eval(new_entry))
                s_idx = self.rng.randint(
                    low=0, high=max(1, len(new_toks) - remaining_toks_cnt + 1)
                )
                additional_toks = new_toks[s_idx:s_idx+remaining_toks_cnt] # TODO(sami) maybe pick from another random example
                seqlens.append(len(additional_toks))
                toks = torch.cat([toks, additional_toks], dim=0)
                remaining_toks_cnt = self.max_seq_length - len(toks)
            assert len(toks) == self.max_seq_length, f"{len(toks)} != {self.max_seq_length}"
            labels = toks.clone()
            return {"input_ids": toks.type(torch.int64), "labels": labels.type(torch.int64), "seqlens": seqlens} 
        
class FakeDataset(Dataset):
    max_len = 1000000

    def __init__(
        self,
        max_seq_length: int = -1,
        context_stuffing: bool = False,
    ) -> None:
        self.max_seq_length = max_seq_length
        assert self.max_seq_length % 2 == 0, "max_seq_length must be even"
        self.context_stuffing = context_stuffing

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        toks = torch.randint(low=0, high=100, size=(self.max_seq_length,), dtype=torch.int64)  # Adjusted to specify range and size
        labels = toks.clone()
        if self.context_stuffing:
            return {"input_ids": toks, "labels": labels, "seqlens": [self.max_seq_length//2, self.max_seq_length//2]}
        else:
            return {"input_ids": toks, "labels": labels}

    def __len__(self):
        return self.max_len

def get_context_stuffing_collate_fn(max_seq_length: int = -1):
    """Returns the collate function for context stuffing pretraining (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_context_stuffing_collate_fn, max_seq_length=max_seq_length)


def _context_stuffing_collate_fn(samples: List[Dict[str, torch.Tensor]], max_seq_length: int = -1) -> Dict[str, torch.Tensor]:
    batched = {}
    for key in ("input_ids", "labels"):
        batched[key] = torch.stack([sample[key] for sample in samples])
        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    batched["seqlens"] =  [x for sample in samples for x in sample["seqlens"]]
    return batched


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
    download_dir: Path = Path("./data/nao_mosaic")
    """The directory in which the downloaded dataset gets saved."""

    local_cache: Path = Path("/tmp/mds-cache/")

    # data_path: Union[str, Path] = Path("data/")
    # """The path to the data directory, containing two folders 'slimpajama' and 'starcoder'
    # which are the output of the preprocessing step done in advance. See the `tutorial/pretrain_tinyllama.md`
    # for instructions. The path can also be a remote path (e.g., s3://)."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    deduplication: bool = True
    # collect_human_virus: bool = True
    collect_human_virus: bool = False
    context_stuffing: bool = False
    fake_data: bool = False


    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = -1 if max_seq_length is None else max_seq_length
    
    def setup(self, rank) -> None:
        if not self.fake_data:
            nao_file_list = [
                "MJ-2024-04-04-44_2-27_S5_L001.collapsed.gz",
                "MJ-2024-02-08-44_Ceres_1_9_S9.collapsed.gz",
                "JR-2024-04-15-nR346G1-P001-L001.collapsed.gz",
                "JR-2024-03-22-a-nR342-L4-G1-P001.collapsed.gz",
                "MJ-2024-04-04-44_2-27_S5_L002.collapsed.gz",
            ]
            
            rank_id = f"rank_{rank}_id"

            stream_list = []
            for nao_file in nao_file_list:
                stream = Stream(
                    remote = f"s3://mgfm-bucket-01/streams/stream_{nao_file}",
                    local = f"/tmp/mds-cache/stream_{nao_file}_{rank_id}",
                    repeat = 1,
                )
                stream_list.append(stream)

            streaming.base.util.clean_stale_shared_memory()
            self.train_dataset = NAODataset(
                batch_size=self.batch_size,
                streams = stream_list[:-1],
                streaming_kwargs = {"shuffle": True},
                tokenizer=self.tokenizer,
                max_seq_length=self.seq_length,
                ignore_index=self.ignore_index,
                context_stuffing=self.context_stuffing,
            )

            self.test_dataset = NAODataset(
                batch_size=self.batch_size,
                streams = stream_list[-1:], # using final stream in list as a validation set
                streaming_kwargs = {"shuffle": True},
                tokenizer=self.tokenizer,
                max_seq_length=self.seq_length,
                ignore_index=self.ignore_index,
                context_stuffing=self.context_stuffing,
            )
        else:
            self.train_dataset = FakeDataset(
                max_seq_length=self.seq_length,
                context_stuffing=self.context_stuffing,
            )
            self.test_dataset = FakeDataset(
                max_seq_length=self.seq_length,
                context_stuffing=self.context_stuffing,
            )

    def get_collate_fn(self):
        if not self.context_stuffing:
            return get_sft_collate_fn(
                max_seq_length=self.seq_length, 
                ignore_index=self.ignore_index, 
                pad_id=self.tokenizer.processor.pad_token_id,
            )
        else:
            return get_context_stuffing_collate_fn(max_seq_length=self.seq_length)


    def train_dataloader(self) -> DataLoader:
        if not self.fake_data:
            return StreamingDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                    collate_fn=self.get_collate_fn()
                )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.get_collate_fn()
            )

    def val_dataloader(self) -> StreamingDataLoader:
        if not self.fake_data:
            return StreamingDataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.get_collate_fn()
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.get_collate_fn()
            )
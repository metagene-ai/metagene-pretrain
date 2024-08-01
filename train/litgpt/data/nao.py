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
        self.bos_token_tensor = torch.tensor([4])
        self.eos_token_tensor = torch.tensor([5])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_len = self.max_seq_length + 1
        # +1 here because we actually want one more token that the max_seq_length because of the target, input
        example = super().__getitem__(idx)["token_ids"]
        toks = torch.tensor(ast.literal_eval(example))
        toks = toks[:max_len-1]
        toks = torch.cat([toks, self.eos_token_tensor], dim=0)
        toks = toks - 1
        if not self.context_stuffing:
            labels = toks.clone()
            return {"input_ids": toks.type(torch.int64), "labels": labels.type(torch.int64)}
        else:
            seqlens = []
            remaining_toks_cnt = max_len - len(toks)
            seqlens.append(len(toks))
            idx_offset = 0
            while remaining_toks_cnt > 0:
                idx_offset += 1
                opposed_idx = len(self) - idx_offset * self.batch_size - idx
                # here we want to get a new index for the additional data. 
                # there is two constraints on the idx:
                # 1. it should be relativly far from the current index to avoid picking a sample that will be consume in a few steps or by another data rank
                # 2. The new idx should be relativlty packed, and not random, otherwise we won't leverage the streaming feature that only download and cache part of the data
                # 
                # a solution is to consume the dataset in reverse when getting new samples.
                # the idx_offset is here in the case of two consective samples beeing still smaller than the seq_len, which should almost never happen
                try:
                    new_entry = super().__getitem__(opposed_idx)["token_ids"]
                except ValueError:
                    s_idx = self.rng.randint(low=0, high=10000)
                    new_entry = super().__getitem__(s_idx)["token_ids"]

                new_toks = torch.tensor(ast.literal_eval(new_entry))
                new_toks = new_toks[:remaining_toks_cnt-1]
                new_toks = torch.cat([new_toks, self.eos_token_tensor], dim=0)
                new_toks = new_toks - 1
                seqlens.append(len(new_toks))
                toks = torch.cat([toks, new_toks], dim=0)
                remaining_toks_cnt = max_len - len(toks)

            # Reduce recorded size of final packed sequence by 1
            seqlens[-1] = seqlens[-1] - 1

            assert len(toks) == max_len, f"{len(toks)} != {max_len}"
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
        max_len = self.max_seq_length + 1
        # +1 here because we actually want one more token that the max_seq_length because of the target, input
        toks = torch.randint(low=0, high=100, size=(max_len,), dtype=torch.int64)  # Adjusted to specify range and size
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
            batched[key] = batched[key][:, :max_seq_length+1] # +1 here because we actually want one more token that the max_seq_length because of the target, input
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
            # rank_id = f"rank_{rank}_id"
            streaming.base.util.clean_stale_shared_memory()

            all_stream = Stream(
                remote = f"s3://mgfm-bucket-01/streams",
                local = f"/tmp/mds-cache/train",
                repeat = 1,
            )
            val_stream = Stream(
                remote = f"s3://mgfm-bucket-01/streams/stream_MJ-2024-04-04-44_2-27_S5_L002.collapsed.gz_small",
                local = f"/tmp/mds-cache/val",
                repeat = 1,
            )
            stream_list = [all_stream, val_stream]

            self.train_dataset = NAODataset(
                batch_size=self.batch_size,
                streams = stream_list[:-1],
                streaming_kwargs = {"shuffle": True, "shuffle_block_size": 50_000_000, "num_canonical_nodes": 1},
                tokenizer=self.tokenizer,
                max_seq_length=self.seq_length,
                ignore_index=self.ignore_index,
                context_stuffing=self.context_stuffing,
            )

            self.test_dataset = NAODataset(
                batch_size=self.batch_size,
                streams = stream_list[-1:], # using final stream in list as a validation set
                streaming_kwargs = {"shuffle": True, "shuffle_block_size": 50_000_000, "num_canonical_nodes": 1},
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

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
import numpy as np
from streaming import StreamingDataset, StreamingDataLoader
import streaming
from streaming.base.stream import Stream

import torch

from litgpt import Tokenizer
# TODO: potentially implement MLMDataset
from litgpt.data import DataModule
from litgpt.data.base import get_sft_collate_fn



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
        example = super().__getitem__(idx)["text"]
        toks = self.tokenizer.encode(example, max_length=self.max_seq_length)
        if not self.context_stuffing:
            labels = toks.clone()
            return {"input_ids": toks.type(torch.int64), "labels": labels.type(torch.int64)}
        else:
            remaining_toks_cnt = self.max_seq_length - len(toks)
            og_toks_len = len(toks)
            if remaining_toks_cnt:
                s_idx = self.rng.randint(
                    low=0, high=max(1, len(toks) - remaining_toks_cnt + 1)
                )
                additional_toks = toks[s_idx:s_idx+remaining_toks_cnt] # TODO(sami) maybe pick from another random example
                toks = torch.cat([toks, additional_toks], dim=0)
                labels = toks.clone()
                seqlens = [og_toks_len, remaining_toks_cnt]

            return {"input_ids": toks.type(torch.int64), "labels": labels.type(torch.int64), "seqlens": seqlens} 


    
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

    nonc_cu_seqlens = [seqlen for sample in samples for seqlen in sample["seqlens"]]
    batched["cu_seqlens"] = _get_cu_seqlens(nonc_cu_seqlens=nonc_cu_seqlens)
    return batched

def _get_cu_seqlens(nonc_cu_seqlens: List[torch.Tensor]) -> torch.Tensor:
    cu_seqlens = []
    running_sum = 0
    for seqlen in nonc_cu_seqlens:
        cu_seqlens.append(seqlen + running_sum)
        running_sum += seqlen

    return torch.Tensor(cu_seqlens).to(torch.int32)


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
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[NAODataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[NAODataset] = field(default=None, init=False, repr=False)
    deduplication: bool = True
    # collect_human_virus: bool = True
    collect_human_virus: bool = False
    context_stuffing: bool = False


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
    
    def setup(self, rank) -> None:

        streaming.base.util.clean_stale_shared_memory()
        rank_id = f"rank_{rank}_id"
        self.train_dataset = NAODataset(
            batch_size=self.batch_size,
            local=str(self.local_cache/"train"/rank_id),
            # local=str(self.local_cache),
            remote=str(self.download_dir),
            split="train",
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            ignore_index=self.ignore_index,
            context_stuffing=self.context_stuffing,
        )

        self.test_dataset = NAODataset(
            batch_size=self.batch_size,
            local=str(self.local_cache/"test"/rank_id),
            split="test",
            remote=str(self.download_dir),
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            ignore_index=self.ignore_index,
            context_stuffing=self.context_stuffing,
        )

    def get_collate_fn(self):
        if not self.context_stuffing:
            return get_sft_collate_fn(
                max_seq_length=self.max_seq_length, 
                ignore_index=self.ignore_index, 
                pad_id=self.tokenizer.processor.pad_token_id,
            )
        else:
            return get_context_stuffing_collate_fn(
                max_seq_length=self.max_seq_length, 
                ignore_index=self.ignore_index, 
            
            )
    def train_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn()
        )

    def val_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.get_collate_fn()
            )
        

        # if self.human_virus_dataset is not None:
        #     dataloaders.append(
        #         DataLoader(
        #             self.human_virus_dataset,
        #             batch_size=self.batch_size,
        #             shuffle=False,
        #             num_workers=self.num_workers,
        #             collate_fn=get_sft_collate_fn(
        #                 max_seq_length=self.max_seq_length, 
        #                 ignore_index=self.ignore_index,
        #                 pad_id=self.tokenizer.processor.pad_token_id,
        #             ),
        #         )
            
        #     )
        # return dataloaders



# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import ast

from streaming import StreamingDataset, StreamingDataLoader
import streaming
from streaming.base.stream import Stream

import torch

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

    ) -> None:
        super().__init__(batch_size=batch_size, streams=streams, remote=remote, local=local, split=split, **streaming_kwargs)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = super().__getitem__(idx)["token_ids"]
        encoded_prompt = torch.tensor(ast.literal_eval(example))
        labels = encoded_prompt.clone()
        return {"input_ids": encoded_prompt.type(torch.int64), "labels": labels.type(torch.int64)}




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
            max_seq_length=self.max_seq_length,
            ignore_index=self.ignore_index,
        )

        self.test_dataset = NAODataset(
            batch_size=self.batch_size,
            streams = stream_list[-1:], # using final stream in list as a validation set
            streaming_kwargs = {"shuffle": True},
            tokenizer=self.tokenizer,
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
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, 
                ignore_index=self.ignore_index, 
                pad_id=self.tokenizer.processor.pad_token_id,
            ),
        )

    def val_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
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



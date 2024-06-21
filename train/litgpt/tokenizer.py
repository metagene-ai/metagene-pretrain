# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import sys
import os
import ipdb
import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

dirname = os.path.dirname(__file__)
pathname = os.path.join(dirname, '../')
sys.path.append(pathname)
from minbpe.minbpe import RegexTokenizer

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(
                f"The checkpoint directory does not exist: {str(checkpoint_dir)}"
            )

        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.use_eos = self.check_if_eos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.model` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()

        elif (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (
                special_tokens_path := checkpoint_dir / "tokenizer_config.json"
            ).is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                self.bos_id = (
                    self.token_to_id(bos_token) if bos_token is not None else None
                )
                eos_token = config.get("eos_token")
                self.eos_id = (
                    self.token_to_id(eos_token) if eos_token is not None else None
                )
            if (
                special_tokens_path := checkpoint_dir / "generation_config.json"
            ).is_file():
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
        elif "genomics" in str(checkpoint_dir):
            model_max_length = 512  # TODO: make this a parameter
            self.processor = CharacterTokenizer(
                characters="ACGTN", model_max_length=model_max_length
            )
            self.backend = "huggingface"
            self.bos_id = self.processor.bos_token_id
            self.eos_id = self.processor.eos_token_id
        elif "minbpe" in str(checkpoint_dir):
            self.backend = "bpe"
            model_max_length = 512  # TODO: make this a parameter
            self.processor = RegexTokenizer(SPLIT_PATTERN)
            # self.processor.load('minbpe/tokenizer/mgfm-1024.model')
            self.processor.load('minbpe/tokenizer/large-mgfm-1024.model')
            vocab_size = self.processor.vocab_size
            self.bos_id = vocab_size - 1
            self.eos_id = vocab_size - 1
            self.processor.pad_token_id = self.bos_id
        else:
            raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            try:
                return self.processor.get_vocab_size(with_added_tokens=False)
            except AttributeError:
                return self.processor.vocab_size
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        if self.backend == "bpe":
            return self.processor.vocab_size
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        elif self.backend == "bpe":
            id_ = self.processor.encode(token)[0]
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if "genomics" in str(checkpoint_dir) or "minbpe" in str(checkpoint_dir):
            return True
        if not (
            tokenizer_config_path := checkpoint_dir / "tokenizer_config.json"
        ).is_file():
            return False
        with open(tokenizer_config_path) as fp:
            config = json.load(fp)
        if any(
            config.get(check, False) for check in ("add_bos_token", "add_prefix_space")
        ):
            return True
        # for examples that also use the Llama tokenizer, but do not have or set add_bos_token to True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return (
            config.get("add_bos_token") is None
            and config.get("tokenizer_class") == "LlamaTokenizer"
        )

    def check_if_eos_token_used(self, checkpoint_dir: Path) -> bool:
        if "genomics" in str(checkpoint_dir) or "minbpe" in str(checkpoint_dir):
            return True
        return False

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string)
            tokens = getattr(tokens, "ids", tokens)
        elif self.backend in ["sentencepiece", "bpe"]:
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        if bos or (bos is None and self.use_bos):
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError(
                    "This tokenizer does not have a defined a bos token"
                )
            if tokens[0] != bos_id:
                tokens = [bos_id] + tokens
        if eos or self.use_eos:
            if tokens[-1] != self.eos_id:
                tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)


# Adapted from HyenaDNA: https://github.com/HazyResearch/hyena-dna/blob/d553021b483b82980aa4b868b37ec2d4332e198a/standalone_hyenadna.py#L939
class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "right",
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=4. Following are list of all of the special tokens with
                their corresponding ids:
                    "[PAD]": 0
                    "[BOS]": 1
                    "[EOS]": 2
                    "[MASK]": 3
                    "[UNK]": 4
                an id (starting at 5) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
            **{ch: i + 5 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def token_to_id(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def get_vocab(self):
        return {}

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    
if __name__ == "__main__":
    tokenizer = Tokenizer(checkpoint_dir="checkpoints/genomics-llama")

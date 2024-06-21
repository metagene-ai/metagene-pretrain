# %%
from minbpe import RegexTokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
tokenizer = RegexTokenizer(GPT2_SPLIT_PATTERN)
tokenizer.load('tokenizer/mgfm.model')
# %%
tokenizer.encode('A')
# %%
tokenizer.decode([65])
# %%
tokenizer.special_tokens
# %%
tokenizer.decode([tokenizer.vocab_size-1])

# %%

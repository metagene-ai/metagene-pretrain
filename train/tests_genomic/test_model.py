import pytest
import torch

from litgpt.model import Config
from litgpt.model import GPT
from lightning.fabric import Fabric

@pytest.fixture
def config() -> Config:
    return Config(
        name="llama",
        n_embd=64,
        n_head=2,
        n_layer=2,
        vocab_size=1024,
    )

@pytest.mark.parametrize("attention_impl", ["sdpa", "fa2"])
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_gpt(config: Config, attention_impl: str, precision: str):
    config.attention_impl = attention_impl
    _test_gpt(config, precision)

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_context_stuffing(config: Config, precision: str):
    config.attention_impl = "fa2"
    config.context_stuffing = True
    _test_gpt(config, precision)


def _test_gpt(config: Config, precision: str):

    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = GPT(config)
    model = fabric.setup(model)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(fabric.device)

    if config.context_stuffing:
        cu_seqlens = torch.randint(0, SEQ_LEN, (BATCH_SIZE,))
        for i in range(1, BATCH_SIZE): # this ensure that cu_seqlens is cummulative, i.e, increasing for each sample in the batch
            cu_seqlens[i] = cu_seqlens[i-1] + cu_seqlens[i]
        cu_seqlens = cu_seqlens.to(torch.int32).to(fabric.device)
    else:
        cu_seqlens = None

    output = model(input, cu_seqlens=cu_seqlens)
    assert output is not None




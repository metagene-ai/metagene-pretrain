from einops import rearrange
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

@pytest.mark.parametrize("attention_impl", ["sdpa","xformers"])
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_gpt(config: Config, attention_impl: str, precision: str):
    config.attention_impl = attention_impl
    _test_gpt(config, precision)

# @pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
# def test_context_stuffing(config: Config, precision: str):
#     config.attention_impl = "xformers"
#     _test_gpt(config, precision, context_stuffing=True)


def _test_gpt(config: Config, precision: str, context_stuffing: bool = False):

    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = GPT(config)
    model = fabric.setup(model)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(fabric.device)

    output = model(input)
    
    assert output is not None
    assert not output.isnan().any()
    
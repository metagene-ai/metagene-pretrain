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

    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = GPT(config)
    model = fabric.setup(model)


    input = torch.randint(0, 1024, (1, 10)).to(fabric.device)

    output = model(input)
    assert output is not None
    




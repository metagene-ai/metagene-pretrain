import pytest
import torch

from litgpt.model import Config
from litgpt.model import GPT

@pytest.fixture
def config() -> Config:
    return Config(
        name="llama",
        n_embd=64,
        n_head=2,
        n_layer=2,
        vocab_size=1024,
    )

def test_gpt(config: Config):
    model = GPT(config)

    input = torch.randint(0, 1024, (1, 10))

    output = model(input)
    assert output is not None
    




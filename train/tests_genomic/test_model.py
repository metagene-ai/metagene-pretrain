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

@pytest.mark.parametrize("attention_impl", ["sdpa", "fa2"])
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_gpt(config: Config, attention_impl: str, precision: str):
    config.attention_impl = attention_impl
    _test_gpt(config, precision)

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_context_stuffing(config: Config, precision: str):
    config.attention_impl = "fa2"
    _test_gpt(config, precision, context_stuffing=True)


def _test_gpt(config: Config, precision: str, context_stuffing: bool = False):

    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = GPT(config)
    model = fabric.setup(model)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE + 1, SEQ_LEN)).to(fabric.device)

    input = batch[:-1]
    target = batch[1:]


    if context_stuffing:
        cu_seqlens = torch.Tensor([i*SEQ_LEN // 2 for i in range(BATCH_SIZE)]).to(torch.int32)
        cu_seqlens = cu_seqlens.to(fabric.device)
        print(cu_seqlens)
    else:
        cu_seqlens = None

    output = model(input, cu_seqlens=cu_seqlens)
    
    print(output.shape)
    assert output is not None

    flatten_logits = rearrange(output, "b seq vocab -> (b seq) vocab")
    flatten_target = rearrange(target, "b seq -> (b seq)")

    loss = torch.nn.functional.cross_entropy(flatten_logits, flatten_target)

    assert not loss.isnan().any()
    # print(f"loss {loss}")   




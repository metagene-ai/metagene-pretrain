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

@pytest.mark.parametrize("attention_impl", ["sdpa", "fa2","xformers"])
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

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(fabric.device)

    if context_stuffing:
        cu_seqlens = torch.Tensor([i*SEQ_LEN // 2 for i in range(BATCH_SIZE)]).to(torch.int32)
        cu_seqlens = cu_seqlens.to(fabric.device)
    else:
        cu_seqlens = None

    output = model(input, cu_seqlens=cu_seqlens)
    
    assert output is not None
    assert not output.isnan().any()
    

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_context_stuffing_output(config: Config, precision: str):
    """
    This test try to asses if context stuffing is producing the good attention output.
    Basically what it does is doing the attention over two samples. Once with the two samples
    concatenated and once with each sample separately. The two sets of attention should be the same.
    """

    config.attention_impl = "fa2"


    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = GPT(config)
    model = fabric.setup(model)

    BATCH_SIZE = 2
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(fabric.device)
    output = model(input)

    ### same with context stuffing
    # instead of being [[0,1,2], [0,3,5]] it will be [[0,1,2,0,3,5]] with cu_seqlens = [3,6]
    input_packed = rearrange(input, "b seq -> 1 (b seq)")
    cu_seqlens = torch.Tensor([SEQ_LEN, 2*SEQ_LEN]).to(torch.int32).to(fabric.device)
    # cu_seqlens = torch.Tensor([7, 16]).to(torch.int32).to(fabric.device)

    model.config.context_stuffing = True
    output_packed = model(input_packed, cu_seqlens=cu_seqlens)

    output_packed = output.reshape(output.shape)
    assert not output_packed.isnan().any()
    assert torch.allclose(output, output_packed)


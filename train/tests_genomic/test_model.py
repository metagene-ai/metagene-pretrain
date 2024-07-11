from typing import Tuple
from einops import rearrange
import pytest
import torch

from litgpt.model import CausalSelfAttention, Config, build_rope_cache
from litgpt.model import GPT
from lightning.fabric import Fabric
from xformers.ops.fmha.common import AttentionFwOpBase

@pytest.fixture
def config() -> Config:
    return Config(
        name="llama",
        n_embd=64,
        n_head=2,
        n_layer=2,
        vocab_size=1024,
    )


PRECISION_TO_DTYPE = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
}

@pytest.mark.parametrize("attention_impl", ["sdpa","xformers"])
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_gpt(config: Config, attention_impl: str, precision: str):
    config.attention_impl = attention_impl
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

    output = model(input)
    
    assert output is not None
    assert not output.isnan().any()

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_gpt_output(config: Config, precision: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers
    """
    
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()


    model = GPT(config)
    model = fabric.setup(model)

    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024

    input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(fabric.device)

    ###  SDPA 
    config.attention_impl = "sdpa"
    output_sdpa = model(input)
    
    ### XFORMERS 
    config.attention_impl = "xformers"
    output_xformers = model(input)

    ### TESTING
    assert output_sdpa.shape == output_xformers.shape
    
    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_sdpa, output_xformers, atol=atol, rtol=rtol)



def get_cos_and_sin_attn(config: Config, seq_len: int, device)-> Tuple[torch.Tensor, torch.Tensor]:
    cos,sin =  build_rope_cache(
        seq_len=seq_len,
        n_elem=config.rope_n_elem,
        device=device,
        condense_ratio=config.rope_condense_ratio,
        base=config.rope_base,
    )
    cos = cos[:seq_len]
    sin = sin[:seq_len] 
    return cos, sin

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_attn_output(config: Config, precision: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers
    """
    
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    model = CausalSelfAttention(config)
    model = fabric.setup(model)

    BATCH_SIZE = 16
    SEQ_LEN = 8

    input = torch.rand(BATCH_SIZE, SEQ_LEN, config.n_embd).to(fabric.device)
    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, fabric.device)
    
    ###  SDPA 
    config.attention_impl = "sdpa"
    output_sdpa = model(input, cos, sin)
    
    ### XFORMERS 
    config.attention_impl = "xformers"
    output_xformers = model(input, cos, sin)

    ### TESTING
    assert output_sdpa.shape == output_xformers.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_sdpa, output_xformers, atol=atol, rtol=rtol)

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
def test_context_stuffing_attn(config: Config, precision: str):
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    config.attention_impl = "xformers"
    model = CausalSelfAttention(config)
    model = fabric.setup(model)

    SEQ_LEN = 8


    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, fabric.device)
    input = torch.rand(2, SEQ_LEN, config.n_embd).to(fabric.device) # [[0,1,2], [1,3, 2]]
    


    ### batch 
    output_xformers = model(input, cos, sin)

    ### context stuffed
    input_context_stuffed = rearrange(input, "b s h -> 1 (b s) h") # [[0, 1, 2, 1, 3, 2]]
    assert input_context_stuffed.shape == (1, 2*SEQ_LEN, config.n_embd)
    seqlens = [SEQ_LEN, SEQ_LEN]

    
    cos, sin = get_cos_and_sin_attn(config, 2*SEQ_LEN, fabric.device)
    output_xformers_context_stuffed = model(input_context_stuffed, cos, sin, seqlens=seqlens)


    output_xformers_context_stuffed = rearrange(output_xformers_context_stuffed, "1 (b s) h -> b s h", b = 2)



    ### TESTING
    assert output_xformers.shape == output_xformers_context_stuffed .shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_xformers, output_xformers_context_stuffed, atol=atol, rtol=rtol)
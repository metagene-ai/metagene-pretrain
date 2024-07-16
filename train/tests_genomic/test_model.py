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

@pytest.mark.parametrize("attention_impl", ["sdpa","xformers", "fa"])
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
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_gpt_output(config: Config, precision: str, attention_impl: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers/fa
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
    config.attention_impl = attention_impl
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
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_attn_output(config: Config, precision: str, attention_impl: str):
    """
    in this test we compare the output of the GPT with sdpa and xformers/fa
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
    config.attention_impl = attention_impl
    output_xformers = model(input, cos, sin)

    ### TESTING
    assert output_sdpa.shape == output_xformers.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_sdpa, output_xformers, atol=atol, rtol=rtol)

@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_context_stuffing_attn(config: Config, precision: str, attention_impl: str):
    """
    In this test we compare normal pad attention with stuffing.

    input is [[2, 1, 4, 8, PAD, PAD, PAD, PAD], [1, 4, 2, 7, PAD, PAD, PAD, PAD]]
    for padded input and [[2, 1, 4, 8, 1, 4, 2, 7]] for stuffed input

    we then compare the output of the two and should be the same.
    """
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    config.attention_impl = attention_impl
    model = CausalSelfAttention(config)
    model = fabric.setup(model)

    SEQ_LEN = 8

    emb = torch.nn.Embedding(10, config.n_embd).to(fabric.device)

    pad_id = 0
    input_raw = torch.Tensor([[2, 1, 4, 8, pad_id, pad_id, pad_id, pad_id], [1, 4, 2, 7, pad_id, pad_id, pad_id, pad_id]]).long().to(fabric.device)
    input = emb(input_raw)

    input_stuff_raw = torch.Tensor([[2, 1, 4, 8, 1, 4, 2, 7]]).long().to(fabric.device)
    seqlens = [4, 4]
    input_stuff = emb(input_stuff_raw)

    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, fabric.device)
        
    ### batch 
    output_ctx_stuff = model(input, cos, sin)
    
    output_ctx_stuff = output_ctx_stuff[:, :4, :] # remove padding token

    output_xformers_stuff = model(input_stuff, cos, sin, seqlens=seqlens)
    output_xformers_stuff = output_xformers_stuff.reshape(2,4,config.n_embd)

    ### TESTING
    assert output_ctx_stuff.shape == output_xformers_stuff.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_ctx_stuff, output_xformers_stuff, atol=atol, rtol=rtol)


@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_context_stuffing_attn_2(config: Config, precision: str, attention_impl: str):
    """
    this test is slightu different from the one above, it tests 
    that passing two time the same input in a stuff way yield the same results.
    """
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    config.attention_impl = attention_impl
    model = CausalSelfAttention(config)
    model = fabric.setup(model)

    SEQ_LEN = 8

    emb = torch.nn.Embedding(10, config.n_embd).to(fabric.device)

    seq = [2,1,4,8] 
    input_stuff_raw = torch.Tensor([seq + seq]).long().to(fabric.device)
    seqlens = [len(seq), len(seq)]
    input_stuff = emb(input_stuff_raw)


    cos, sin = get_cos_and_sin_attn(config, SEQ_LEN, fabric.device)

    output = model(input_stuff, cos, sin, seqlens=seqlens)
    
    output_left = output[:, :4, :]
    output_right = output[:, 4:, :]

    ### TESTING
    assert output_left.shape == output_right.shape

    ### xformers has a higher tolerance
    atol = AttentionFwOpBase.ERROR_ATOL[PRECISION_TO_DTYPE[precision]]
    rtol = AttentionFwOpBase.ERROR_RTOL[PRECISION_TO_DTYPE[precision]]
    torch.testing.assert_close(output_left, output_right, atol=atol, rtol=rtol)


@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
@pytest.mark.parametrize("attention_impl", ["xformers", "fa"])
def test_context_stuffing_backward(config: Config, precision: str, attention_impl: str):
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    fabric.launch()

    config.attention_impl = attention_impl
    model = GPT(config)
    model = fabric.setup(model)


    BATCH_SIZE = 16
    SEQ_LEN = 8
    VOCAB_SIZE = 1024
    ITER = 5


    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE+1, SEQ_LEN)).to(fabric.device)

    input = batch[:-1]
    target = batch[1:]

    seqlens = [SEQ_LEN//2 for _ in range(2*BATCH_SIZE)]


    for _ in range(ITER):

        output = model(input, seqlens=seqlens)

        assert output is not None
        assert not output.isnan().any()

        flatten_logits = rearrange(output, "b seq vocab -> (b seq) vocab")
        flatten_target = rearrange(target, "b seq -> (b seq)")

        loss = torch.nn.functional.cross_entropy(flatten_logits, flatten_target)
        fabric.backward(loss)

        assert not loss.isnan().any()


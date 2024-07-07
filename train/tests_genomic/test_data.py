import torch
from litgpt.data.nao import get_context_stuffing_collate_fn, _get_cu_seqlens


def test_collate_fn():
    max_seq_length = 16
    collate_fn = get_context_stuffing_collate_fn(max_seq_length=max_seq_length)
    samples = []

    non_cu_seqlens = [
        [9, 7],
        [15, 1],
        [16],
        [3, 13],
        [8, 5, 3],
    ]

    for seqlen in non_cu_seqlens:
        input_ids = torch.randint(0, 10, (max_seq_length,))
        labels = input_ids.clone()
        samples.append({"input_ids": input_ids,"labels": labels, "seqlens": seqlen})

    batch = collate_fn(samples)
    
    assert batch["input_ids"].shape == (len(samples), max_seq_length)
    assert batch["labels"].shape == (len(samples), max_seq_length)

    assert -100 not in batch["labels"].tolist() # no padding tokens with context stuffing

    assert len(batch["cu_seqlens"]) == len([seqlen for sample in non_cu_seqlens for seqlen in sample])

    assert batch["cu_seqlens"].tolist() == [9, 16, 31, 32, 48, 51, 64, 72, 77, 80]



def test_get_cu_seqlens():
    non_cu_seq_lens = [3, 4 , 8 ,1 ,12]
    cu_seq_lens = _get_cu_seqlens(non_cu_seq_lens)
    assert cu_seq_lens.tolist() == [3, 7, 15, 16, 28] 
    

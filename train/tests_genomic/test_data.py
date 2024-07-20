import torch
from litgpt.data.nao import get_context_stuffing_collate_fn


def test_collate_fn():
    max_seq_length = 16
    collate_fn = get_context_stuffing_collate_fn(max_seq_length=max_seq_length)
    samples = []

    seqlens = [
        [9, 7],
        [15, 1],
        [16],
        [3, 13],
        [8, 5, 3],
    ]

    for seqlen in seqlens:
        input_ids = torch.randint(0, 10, (max_seq_length,))
        labels = input_ids.clone()
        samples.append({"input_ids": input_ids,"labels": labels, "seqlens": seqlen})

    batch = collate_fn(samples)
    
    assert batch["input_ids"].shape == (len(samples), max_seq_length)
    assert batch["labels"].shape == (len(samples), max_seq_length)

    assert -100 not in batch["labels"].tolist() # no padding tokens with context stuffing

    assert len(batch["seqlens"]) == len([seqlen for sample in seqlens for seqlen in sample])

    assert batch["seqlens"] == [x for sublist in seqlens for x in sublist]

    

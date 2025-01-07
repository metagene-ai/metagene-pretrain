# METAGENE-1 Pretraining

This repository contains code for pretraining METAGENE-1: Metagenomic Foundation Model.

## Introduction

METAGENE-1 is a 7-billion-parameter autoregressive transformer language model, which we
refer to as a metagenomic foundation model, pretrained on a novel corpus of diverse
metagenomic DNA and RNA reads sequenced from wastewater.

This repository contains code for pretraining METAGENE-1. It aims to provide a reference
for future pretraining efforts. Note that the metagenomic pretraining dataset is not yet
public (see [data details](#data-details) below). However, this repository will be
updated in the future as the metagenomic data is publicly released.

> [!NOTE]  
> This repository is still being organized for release, but is provided as a helpful
> reference in the meantime.

## Quick Tour

- [`train`](train/): pretraining code for METAGENE-1.
    - Data
        - [`base.py`](train/litgpt/data/base.py): containing dataset class, NAODataset.
        - [`nao.py`](train/litgpt/data/nao.py): containing NAO (DataModule) class, with
          train/val dataloaders.
    - Model
        - [`genomicsllama.yml`](train/config_hub/pretrain/genomicsllama.yml): pretrain
          configuration for genomics-llama.
        - [`config`](train/litgpt/config.py): model configuration for genomics_llama.
    - Train
        - [`pretrain.py`](train/litgpt/pretrain.py): pretraining model, e.g., initialize
          weights, setup optimizers, setup dataloaders, setup fabric, run training.

## Installation

To install dependencies, run:

```bash
cd train
pip install -e .
pip install -e '.[all]'
pip install -r minbpe/requirements.txt
```

## Pretraining 

Before running pretraining, you will need to update lines in `train/litgpt/data/nao.py`
to point to an s3 bucket containing tokenized metagenomic data files in MosaicML
streaming format.

And then run:
```bash
cd train
python litgpt/pretrain.py --config config_hub/pretrain/genomicsllama.yml
```

Or, as an example with more configurations shown:
```bash
cd train
python litgpt/pretrain.py --config config_hub/pretrain/genomicsllama.yml --fsdp_strategy
_HYBRID_SHARD_ZERO2 --model_name genomics-llama-7b --attention_impl fa --fake_data False
--train.log_stability_interval 25 --eval.interval 25 --train.save_interval 500
```

## Data Details

METAGENE-1 is trained on a newly collected metagenomic dataset comprising material from
a very broad range (e.g., tens of thousands) of organisms, which was collected via
metagenomic sequencing of human wastewater (i.e., municipal influent). This approach
contrasts with prior genomic sequence models, which often focus on curated collections
of specific species or genomic types.

In this implementation, data for pretraining is streamed from a given s3 bucket using
MosaicML's [streaming](https://github.com/mosaicml/streaming) package. If you want to
use this code as-is, you will need to specify an s3 bucket containing MosaicML
streaming-compatible data [here](train/litgpt/data/nao.py#L196) and
[here](train/litgpt/data/nao.py#L201).

> [!NOTE]  
> The metagenomic pretraining dataset is not yet available for public release, but will
> be publicly available in the coming months. When it is available, we will update this
> section of the README with details.

## Byte-pair Encoding (BPE) Tokenization

We trained a BPE tokenizer on ~150M sequence reads (2B base pairs) sampled uniformly at
random from our full set of data files.  Some examples from our vocabulary are listed
below.

A few short tokens:
```
· AA
· GG
· TAC
· AAAA
· ACCC
· ATCC
· TTCC
· AGCC
```

A few longer tokens:
```
· ATTTCACCGC
· TGCCTCCCGTAGG
· TCATTATGCAAAAGGC
· GTATTACCGCGGCTGCTGGC
· ACTACCAGGGTATCTAATCCTGTT
· ACCGTTGCCGGCGTACTCCCCAGGTGGATAGCTTAATGGTTTCCCTCAGGCACCC
```

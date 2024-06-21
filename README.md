# Metagenomic Foundation Model for Pandemic Monitoring

## Overview

We are a team of researchers from USC Computer Science, collaborating with the Nucleic
Acid Observatory (NAO) to pre-train an autoregressive foundation model on metagenomic
sequencing data, given large quantities of reads collected from human wastewater samples
using massively parallel sequencing. This model will be used for embedding, matching,
search, and anomaly detection, with the goal of supporting online pandemic and pathogen
monitoring.

## Goals and Status

**Goal**: Our goal is to train a ~7B parameter metagenomic sequence model following
similar implementation, data scale, and techniques as modern LLMs.

**Status**: We have so far been given ~2T+ tokens of real metagenomic sequence data (see
[data details](#data-details)), collected by multiple groups over different dates and
locations.
- We've trained a 200M parameter and 800M parameter model on a small subset of data
  using single-character (base-pair) tokenization.
- Ran evaluation on perplexity of held-out data, and on human-infecting virus subset.
- Did small study on overfitting on multiple epochs through a chunk of the data.
- Explored other tokenization schemes (k-mer, codons, BPE, etc).
- Trained a byte-pair encoding (BPE) tokenizer on uniformly sampled reads from full data
  (see [details](#byte-pair-encoding-bpe-tokenization)).
- Trained an 800M parameter model on a small subset of data using BPE tokenization.

**Next Steps**: Finish final BPE tokenizer. Tokenize all/portion of data. Train 7B
parameter model on data.

## Quick Tour

- [`tokenize/`](tokenize/): byte-pair encoding (BPE) tokenization of genomic data.
    - [`gather_data.py`](tokenize/gather_data.py): uniformly sample sequence reads for
      BPE tokenization.
    - [`run.py`](tokenize/run.py): run BPE tokenization on sampled sequence reads.
<br/><br/>
- [`train/`](train/): training of the metagenomic foundation model (MGFM).
    - Data
        - [`download.py`](train/download.py): download data files from S3 bucket.
        - [`base.py`](train/litgpt/data/base.py): containing dataset class, NAODataset.
        - [`nao.py`](train/litgpt/data/nao.py): containing NAO (DataModule) class, with
          train/val dataloaders.
    - Model
        - [`genomicsllama.yml`](train/config_hub/pretrain/genomicsllama.yml): pretrain
          configuration for genomics-llama.
        - [`config`](train/litgpt/config.py): model configuration for genomics_llama.
    - Tokenize
        - [`tokenizer.py`](train/litgpt/tokenizer.py): incorporate sequence tokenizer.
    - Train
        - [`pretrain.py`](train/litgpt/pretrain.py): pretraining model, e.g., initialize
          weights, setup optimizers, setup dataloaders, setup fabric, run training.

## Running Model Training

First download a sample of the [wastewater sequence data (1.9 GB)
here](https://drive.google.com/file/d/1hbq0BTS0zbVS8Y708NE4_O21TmRuIM8B/view?usp=drive_link).

Update [line
31](https://github.com/oliu-io/NAOSeq/blob/459b0293380ff88a787cbac3f763011e66dc128f/litgpt/data/nao.py#L31)
in `train/litgpt/data/nao.py` to point to the directory containing the downloaded data.
*Note: this directory should contain only the training data, saved as .txt files*.

Run training:
```bash
cd train
python litgpt/pretrain.py --config config_hub/pretrain/genomicsllama.yml
```

## Data Details

Sequence read data files and approximate numbers of base pairs are listed below (see the
[full list here](data_files.md)).
```
    JR-2024-04-12-nR345P1-L001.collapsed.gz, 40529468973
    JR-2024-04-12-nR345P1-L002.collapsed.gz, 41918174226
    JR-2024-04-12-nR345P1-L003.collapsed.gz, 40823507762
    JR-2024-04-12-nR345P1-L004.collapsed.gz, 41417854382
    JR-2024-04-12-nR345P2-L001.collapsed.gz, 30588984812
    JR-2024-04-12-nR345P2-L002.collapsed.gz, 31358323741
    ...
    ...     [77 total JR files]
    ...
    JR-2024-04-16-nR347G1-P001-L001.collapsed.gz, 26301966946
    JR-2024-04-16-nR347G1-P001-L002.collapsed.gz, 33376000921
    JR-2024-04-16-nR347G1-P002-L001.collapsed.gz, 24448009107
    JR-2024-04-16-nR347G1-P002-L002.collapsed.gz, 31237213961
    JR-2024-04-16-nR347G1-P003-L001.collapsed.gz, 36639747078
    JR-2024-04-16-nR347G1-P003-L002.collapsed.gz, 31556261984
    ·························································
    MJ-2023-12-20-44_24mo_11-21_S4.collapsed.gz, 6424183105
    MJ-2023-12-20-44x10_11-28_S6.collapsed.gz, 9235324835
    MJ-2023-12-20-Ase_11-17_S5.collapsed.gz, 8276253671
    MJ-2023-12-20-CUBE_S3.collapsed.gz, 873402915
    MJ-2023-12-20-WFP_S2.collapsed.gz, 1095931874
    MJ-2024-02-08-44_10x_12_5_reextr_S1.collapsed.gz, 53131314
    ...
    ...     [183 total MJ files]
    ...
    MJ-2024-05-17-44_ActiveCarbon_4-9_S5_L002.collapsed.gz, 43245721732
    MJ-2024-05-17-44_ActiveCarbon_4-9_S5_L003.collapsed.gz, 42920374381
    MJ-2024-05-17-44_ActiveCarbon_4-9_S5_L004.collapsed.gz, 43042590794
    MJ-2024-05-17-44_Torpedo_4-9_S6_L001.collapsed.gz, 37530903387
    MJ-2024-05-17-44_Torpedo_4-9_S6_L002.collapsed.gz, 37582887195
    MJ-2024-05-17-44_Torpedo_4-9_S6_L003.collapsed.gz, 37453956571
```

## Byte-pair Encoding (BPE) Tokenization

We trained a BPE tokenizer on ~150M sequence reads sampled uniformly at random from our
full set of data files.  Some examples from our vocabulary (e.g., size 4096) are listed
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

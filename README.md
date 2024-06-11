# Metagenomic Foundation Model for Pandemic Monitoring

## Overview

We are a team of researchers from USC Computer Science, collaborating with the Nucleic
Acid Observatory (NAO) to pre-train an autoregressive foundation model on metagenomic
sequencing data, given large quantities of reads collected from human wastewater samples
using massively parallel sequencing. This model will be used for embedding, matching,
search, and anomaly detection, with the goal of supporting online pandemic and pathogen
monitoring.

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

## Data Details

Sequence read data files and approximate numbers of base pairs are listed below:
```
    "JR-2024-04-16-nR347G1-P001-L001.collapsed.gz": 27075334829.0,
    "JR-2024-04-16-nR347G1-P001-L002.collapsed.gz": 34357369623.063618,
    "JR-2024-04-16-nR347G1-P002-L001.collapsed.gz": 25166864282.25164,
    "JR-2024-04-16-nR347G1-P002-L002.collapsed.gz": 32155695003.427902,
    "JR-2024-04-16-nR347G1-P003-L001.collapsed.gz": 37717081092.412445,
    "JR-2024-04-16-nR347G1-P003-L002.collapsed.gz": 32484124130.680115,
    "JR-2024-04-16-nR347G1-P004-L001.collapsed.gz": 40316136986.93217,
    "JR-2024-04-16-nR347G1-P004-L002.collapsed.gz": 37032226006.813995,
    "JR-2024-04-16-nR347G1-P005-L001.collapsed.gz": 30543339460.5102,
    "JR-2024-04-16-nR347G1-P005-L002.collapsed.gz": 29885191460.07372,
    "JR-2024-04-16-nR347G1-P006-L001.collapsed.gz": 32283291342.28628,
    "JR-2024-04-16-nR347G1-P006-L002.collapsed.gz": 24310246927.30078,
    "JR-2024-04-15-nR346G1-P001-L001.collapsed.gz": 43116401651.72576,
    "JR-2024-04-15-nR346G1-P001-L002.collapsed.gz": 41816048938.46483,
    "JR-2024-04-15-nR346G1-P001-L003.collapsed.gz": 41740681293.7123,
    "JR-2024-04-15-nR346G1-P002-L001.collapsed.gz": 42252152328.30487,
    "JR-2024-04-15-nR346G1-P002-L002.collapsed.gz": 40966216823.71234,
    "JR-2024-04-15-nR346G1-P002-L003.collapsed.gz": 40878137048.38761,
    "JR-2024-04-15-nR346G1-P003-L001.collapsed.gz": 40973569077.2282,
    "JR-2024-04-15-nR346G1-P003-L002.collapsed.gz": 39816517266.533714,
    "JR-2024-04-15-nR346G1-P003-L003.collapsed.gz": 39538845148.32654,
    "JR-2024-04-15-nR346G1-P004-L001.collapsed.gz": 49590523836.54163,
    "JR-2024-04-15-nR346G1-P004-L002.collapsed.gz": 47499868476.302345,
    "JR-2024-04-15-nR346G1-P004-L003.collapsed.gz": 47855476183.07267,
    "JR-2024-04-15-nR346G1-P005-L001.collapsed.gz": 37520419676.249054,
    "JR-2024-04-15-nR346G1-P005-L002.collapsed.gz": 35835213838.858215,
    "JR-2024-04-15-nR346G1-P005-L003.collapsed.gz": 36161397883.56614,
    "JR-2024-04-15-nR346G1-P006-L001.collapsed.gz": 30784924181.780914,
    "JR-2024-04-15-nR346G1-P006-L002.collapsed.gz": 30168613611.79361,
    "JR-2024-04-15-nR346G1-P006-L003.collapsed.gz": 29928743212.250076,
    "JR-2024-04-15-nR346P1-L004.collapsed.gz": 42712924485.396095,
    "JR-2024-04-15-nR346P2-L004.collapsed.gz": 41447600589.57534,
    "JR-2024-04-15-nR346P3-L004.collapsed.gz": 40283513087.30002,
    "JR-2024-04-15-nR346P4-L004.collapsed.gz": 48609117391.95697,
    "JR-2024-04-15-nR346P5-L004.collapsed.gz": 36462766077.90582,
    "JR-2024-04-15-nR346P6-L004.collapsed.gz": 30348230388.518787,
    "JR-2024-04-15-nR346PrNotRecog-L004.collapsed.gz": 6810434743.618258,
    "JR-2024-04-12-nR345P1-L001.collapsed.gz": 41721174128.06519,
    "JR-2024-04-12-nR345P1-L002.collapsed.gz": 43150712070.854485,
    "JR-2024-04-12-nR345P1-L003.collapsed.gz": 42023858664.312614,
    "JR-2024-04-12-nR345P1-L004.collapsed.gz": 42635681109.90069,
    "JR-2024-04-12-nR345P2-L001.collapsed.gz": 31488405698.30977,
    "JR-2024-04-12-nR345P2-L002.collapsed.gz": 32280365826.034687,
    "JR-2024-04-12-nR345P2-L003.collapsed.gz": 31637270438.779953,
    "JR-2024-04-12-nR345P2-L004.collapsed.gz": 31939574604.800533,
    "JR-2024-04-12-nR345P3-L001.collapsed.gz": 45937337133.38492,
    "JR-2024-04-12-nR345P3-L002.collapsed.gz": 47225612236.43238,
    "JR-2024-04-12-nR345P3-L003.collapsed.gz": 46163231465.52389,
    "JR-2024-04-12-nR345P3-L004.collapsed.gz": 46651797526.17691,
    "JR-2024-04-12-nR345P4-L001.collapsed.gz": 56617842858.097855,
    "JR-2024-04-12-nR345P4-L002.collapsed.gz": 58055321138.27034,
    "JR-2024-04-12-nR345P4-L003.collapsed.gz": 56799798915.50113,
    "JR-2024-04-12-nR345P4-L004.collapsed.gz": 57228029243.692276,
    "JR-2024-04-12-nR345P5-L001.collapsed.gz": 40010325418.52184,
    "JR-2024-04-12-nR345P5-L002.collapsed.gz": 41056427402.977905,
    "JR-2024-04-12-nR345P5-L003.collapsed.gz": 40219781866.10043,
    "JR-2024-04-12-nR345P5-L004.collapsed.gz": 40606828056.813736,
    "JR-2024-04-12-nR345P6-L001.collapsed.gz": 43890467670.645386,
    "JR-2024-04-12-nR345P6-L002.collapsed.gz": 45324096402.21511,
    "JR-2024-04-12-nR345P6-L003.collapsed.gz": 44208662090.13417,
    "JR-2024-04-12-nR345P6-L004.collapsed.gz": 44635785575.66127,
    "JR-2024-04-12-nR345PrNotRecog-L001.collapsed.gz": 7117270580.273461,
    "JR-2024-04-12-nR345PrNotRecog-L002.collapsed.gz": 7150192502.696733,
    "JR-2024-04-12-nR345PrNotRecog-L003.collapsed.gz": 7150553421.522352,
    "JR-2024-04-12-nR345PrNotRecog-L004.collapsed.gz": 7171546202.529167,
    "JR-2024-03-22-b-nR342-L4-G3-P033.collapsed.gz": 3038257087.5389175,
    "JR-2024-03-22-b-nR342-L4-G3-P034.collapsed.gz": 3208118854.6873984,
    "JR-2024-03-22-b-nR342-L4-G3-P035.collapsed.gz": 3190428713.214731,
    "JR-2024-03-22-b-nR342-L4-G3-P036.collapsed.gz": 3499543091.937229,
    "JR-2024-03-22-b-nR342-L4-G3-P037.collapsed.gz": 2799927190.116253,
    "JR-2024-03-22-b-nR342-L4-G3-P038.collapsed.gz": 2452367743.795853,
    "JR-2024-03-22-a-nR342-L4-G1-P001.collapsed.gz": 3049737145.432355,
    "JR-2024-03-22-a-nR342-L4-G1-P002.collapsed.gz": 2591680687.572199,
    "JR-2024-03-22-a-nR342-L4-G1-P003.collapsed.gz": 3730370773.2797093,
    "JR-2024-03-22-a-nR342-L4-G1-P004.collapsed.gz": 4363138832.712662,
    "JR-2024-03-22-a-nR342-L4-G1-P005.collapsed.gz": 3200639764.978654,
    "JR-2024-03-22-a-nR342-L4-G1-P006.collapsed.gz": 3662629895.8334746
```

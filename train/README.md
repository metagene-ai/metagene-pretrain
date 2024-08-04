# Training of the Metagenomic Foundation Model

This directory contains code for (pre-)training of the metagenomic foundation model.

Our code is based on [LitGPT](https://github.com/Lightning-AI/litgpt). See the LitGPT README
[here](README-litgpt.md).

## S3 Scripts

**Selecting or modifying the `index.json` file for pretraining data**

See [scripts/select_training_index_file.sh](scripts/select_training_index_file.sh).

**Uploading checkpoints to S3 bucket**

From within this directory (`train/`), run:
```bash
source scripts/upload_checkpoints_to_s3.sh
```
This assumes the checkpoints are in subdirectories of `out/pretrain/genomics-llama/`.

## Quick Tour

- Data
    - [`download.py`](download.py): download data files from S3 bucket.
    - [`base.py`](litgpt/data/base.py): containing dataset class, NAODataset.
    - [`nao.py`](litgpt/data/nao.py): containing NAO (DataModule) class, with
      train/val dataloaders.
- Model
    - [`genomicsllama.yml`](config_hub/pretrain/genomicsllama.yml): pretrain
      configuration for genomics-llama.
    - [`config`](litgpt/config.py): model configuration for genomics_llama.
- Tokenize
    - [`tokenizer.py`](litgpt/tokenizer.py): incorporate sequence tokenizer.
- Train
    - [`pretrain.py`](litgpt/pretrain.py): pretraining model, e.g., initialize
      weights, setup optimizers, setup dataloaders, setup fabric, run training.

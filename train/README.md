# Training of the Metagenomic Foundation Model

This directory contains code for (pre-)training of the metagenomic foundation model.

Our code is based on [LitGPT](https://github.com/Lightning-AI/litgpt). See the LitGPT README
[here](README-litgpt.md).

## S3 Scripts

### (1) Selecting or modifying the `index.json` file for pretraining data

See [scripts/select_training_index_file.sh](scripts/select_training_index_file.sh).

Note: after we run this script to modify the index.json file on S3, we will need to resume
training with both `--resume <path>` and ``--new_index_file True`` flags.

### (2) Uploading checkpoints to S3 bucket

From within this directory (`train/`), run:
```bash
source scripts/upload_checkpoints_to_s3.sh
```
which will upload all checkpoints to the S3 bucket. This assumes that all checkpoints
are in subdirectories of `out/pretrain/genomics-llama/`.

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

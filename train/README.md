# Training of the Metagenomic Foundation Model

This directory contains code for (pre-)training of the metagenomic foundation model.

Our code is based on [LitGPT](https://github.com/Lightning-AI/litgpt). See the LitGPT README
[here](README-litgpt.md).

## Quick Tour

- Tokenize
    - [`tokenizer.py`](litgpt/tokenizer.py): incorporate sequence tokenizer.
- Data
    - [`base.py`](litgpt/data/base.py): containing dataset class, NAODataset.
    - [`nao.py`](litgpt/data/nao.py): containing NAO (DataModule) class, with
      train/val dataloaders.
- Model
    - [`genomicsllama.yml`](config_hub/pretrain/genomicsllama.yml): pretrain
      configuration for genomics-llama.
    - [`config`](litgpt/config.py): model configuration for genomics_llama.
- Train
    - [`pretrain.py`](litgpt/pretrain.py): pretraining model, e.g., initialize
      weights, setup optimizers, setup dataloaders, setup fabric, run training.

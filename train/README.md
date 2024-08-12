# Training of the Metagenomic Foundation Model

This directory contains code for (pre-)training of the metagenomic foundation model.

Our code is based on [LitGPT](https://github.com/Lightning-AI/litgpt). See the LitGPT README
[here](README-litgpt.md).

## Switching to next pretraining data chunk and resuming training

When we need to transfer from one data chunk to another, we need to do three steps:
(1) Update the `index.json` file on S3 to the index file for the next chunk, (2) clear the streaming cache  (3) run
training, resuming from final checkpoint, with an additional flag.

**First**, to update the `index.json` file on S3, update or uncomment lines in
[scripts/select_training_index_file.sh](scripts/select_training_index_file.sh) to point
to the correct data chunk, and then run:
```bash
source scripts/select_training_index_file.sh
```
**Second**, to clear the streaming cache, run:
```bash
rm -rf /tmp/mds-cache
```
**Third**, to resume training on this new data chunk, run the usual pretraining run
command along with the two flags: `--resume <path>` and ``--new_index_file True``.

Note: using both of these flags is only needed for the "first resume" after switching to
a new index file. If the job needs to be resumed again (from a later checkpoint) before
we've made it to the next index file, we should only use the flag `--resume <path>`
pointing to the later checkpoint.

## Uploading checkpoints to S3 bucket
See [scripts/upload_checkpoints_to_s3.sh](scripts/upload_checkpoints_to_s3.sh) for
details.

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

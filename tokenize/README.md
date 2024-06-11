# Tokenization for the Metagenomic Foundation Model

This directory contains code for tokenization of metagenomic sequence data for the
metagenomic foundation model.

Our code is based on [minbpe](https://github.com/karpathy/minbpe). See the minbpe README
[here](README-minbpe.md).

## Quick Tour

- [`gather_data.py`](gather_data.py): uniformly sample sequence reads for BPE
  tokenization.

- [`run.py`](run.py): run BPE tokenization on sampled sequence reads.

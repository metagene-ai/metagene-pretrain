# Assumes checkpoints are subdirectories of out/pretrain/genomics-llama/

aws s3 sync out/pretrain/genomics-llama s3://mgfm-bucket-01/model-checkpoints/7b

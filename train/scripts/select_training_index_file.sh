# To use the small training data file, run the following command:
# (this has already been run, so the small data file is currently in use).
#aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_10.json s3://mgfm-bucket-01/streams/index.json

# To switch to large training data file (too large/hangs on shuffle), run the following command:
#aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_87.json s3://mgfm-bucket-01/streams/index.json

# To switch to "quarter-sized" training data files, run the following command:
#aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_26a.json s3://mgfm-bucket-01/streams/index.json
aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_26b.json s3://mgfm-bucket-01/streams/index.json
#aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_26c.json s3://mgfm-bucket-01/streams/index.json
#aws s3 cp s3://mgfm-bucket-01/streams/index_files/index_26d.json s3://mgfm-bucket-01/streams/index.json

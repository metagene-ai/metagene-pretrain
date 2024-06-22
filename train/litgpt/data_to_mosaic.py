import numpy as np
from streaming import MDSWriter
from litgpt.utils import CLI
from pathlib import Path
from tqdm import tqdm

def main(
    data_dir = "data/nao_mosaic", # Local or remote directory in which to store the compressed output files
    original_data= Path("data/nao")
):
    # A dictionary mapping input fields to their data types
    columns = {'text': 'str'}

    # Shard compression, if any
    compression = 'zstd'


    print("start to convert data to mosaic")

    # Save the samples as shards using MDSWriter
    with MDSWriter(out=data_dir, columns=columns, compression=compression) as out:
        for fname in tqdm(original_data.glob("*.txt")):
            with open(fname, "r") as f:
                for line in f:
                    sample = {'text': line}
                    out.write(sample)

    print("done")

if __name__ == "__main__":
    CLI(main)


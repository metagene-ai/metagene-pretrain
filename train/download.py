# %%
import os
import argparse
from pathlib import Path

BUCKET = "nao-mgs"
DATA_DIR = Path("/home/oliver/data/")


def parse_directory(run_id):
    s3_path = f"s3://{BUCKET}/{run_id}/cleaned/"
    fps = os.popen(f"aws s3 ls {s3_path}").read().split()
    fps = [fp for fp in fps if fp.endswith(".collapsed.gz")]
    return sorted(fps)


def download_files(run_id, max_files=-1):
    fps = parse_directory(run_id)
    if max_files == -1:
        max_files = len(fps)
    fps = fps[: min(max_files, len(fps))]

    if not Path.exists(DATA_DIR / run_id):
        os.mkdir(DATA_DIR / run_id)

    for fp in fps:
        if Path.exists(DATA_DIR / run_id / fp):
            print(f"{run_id}/{fp} already exists")
        else:
            print(f"Downloading {run_id}/{fp}")
            os.system(
                f"aws s3api get-object --bucket {BUCKET} --key {run_id}/cleaned/{fp} {DATA_DIR / run_id / fp}"
            )
        os.system(f"gunzip {DATA_DIR / run_id / fp}")

    print("Done!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run_id", type=str, required=True)
    args.add_argument("--max_files", type=int, default=-1)
    args = args.parse_args()
    download_files(args.run_id, args.max_files)

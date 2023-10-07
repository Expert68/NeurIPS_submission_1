import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(
    repo_id: Optional[str] = None, access_token: Optional[str] = os.getenv("HF_TOKEN")) -> None:
    if repo_id is None:
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(['Secbone/llama-2-13B-instructed']))
        return

    from huggingface_hub import snapshot_download

    if ("meta-llama" in repo_id or "falcon-180" in repo_id) and not access_token:
        raise ValueError(
            f"{repo_id} requires authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )


    directory = f'llama-33B-instructed'
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=access_token,
    )

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default='Secbone/llama-33B-instructed',help="repo_id")
args = parser.parse_args()


if __name__ == "__main__":
    download_from_hub(args.repo_id)
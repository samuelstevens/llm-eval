import argparse
import os

from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError


def hf_download(repo: str, out: str, hf_token: str | None = None) -> None:
    local_dir = os.path.join(out, repo)
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo, local_dir=local_dir, local_dir_use_symlinks=False, token=hf_token
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass `--hf_token=...` to download private checkpoints.")
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from HuggingFace Hub.")
    parser.add_argument(
        "--repo",
        type=str,
        default="meta-llama/llama-2-7b-chat-hf",
        help="Repository ID to download from.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token."
    )
    parser.add_argument(
        "--out", type=str, default="checkpoints", help="Where to store files."
    )

    args = parser.parse_args()
    hf_download(args.repo, args.out, args.hf_token)

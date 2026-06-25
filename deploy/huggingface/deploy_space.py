"""
deploy/huggingface/deploy_space.py
==================================

One-shot deploy of the FastAPI backend to a Hugging Face Space (Docker SDK).

Hugging Face Spaces gives 16 GB RAM free — enough for TensorFlow, unlike most
free PaaS tiers. This script creates (or reuses) a Docker Space and uploads the
project so the Space builds the repo's Dockerfile.

Usage
-----
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx           # a WRITE token from hf.co/settings/tokens
    python deploy/huggingface/deploy_space.py --owner <hf-username> --name btc-forecasting-api

The dashboard (Vercel) then points at:  https://<owner>-<name>.hf.space
"""

from __future__ import annotations

import argparse
import os
import sys

# Files/dirs that must NOT be uploaded (frontend is deployed separately; venv,
# caches, local MLflow store and git internals are irrelevant to the API image).
IGNORE = [
    "frontend/*", "frontend/**",
    "venv/*", "venv/**", ".git/*", ".git/**",
    "mlruns/*", "mlruns/**", "mlflow.db",
    "**/__pycache__/*", "**/__pycache__/**", "*.pyc",
    ".pytest_cache/*", ".pytest_cache/**",
    ".claude/*", ".claude/**",
    "deploy/huggingface/*",  # the Space README is uploaded explicitly below
    "README.md",  # keep the Space manifest README; don't overwrite with project README
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="Your Hugging Face username/org")
    ap.add_argument("--name", default="btc-forecasting-api", help="Space name")
    ap.add_argument("--private", action="store_true", help="Create a private Space")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN (a write token from hf.co/settings/tokens).")
        return 2

    from huggingface_hub import HfApi, upload_file, upload_folder

    repo_id = f"{args.owner}/{args.name}"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    space_readme = os.path.join(project_root, "deploy", "huggingface", "README.md")

    api = HfApi(token=token)
    print(f"[hf] Creating/own Space {repo_id} (docker) ...")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    print("[hf] Uploading Space manifest (README.md with SDK config) ...")
    upload_file(path_or_fileobj=space_readme, path_in_repo="README.md",
                repo_id=repo_id, repo_type="space", token=token,
                commit_message="Add Space manifest")

    print("[hf] Uploading project (Dockerfile, code, models, artifacts) ...")
    upload_folder(folder_path=project_root, repo_id=repo_id, repo_type="space",
                  token=token, ignore_patterns=IGNORE,
                  commit_message="Deploy BTC forecasting API")

    print(f"\n[hf] Done. Space building at: https://huggingface.co/spaces/{repo_id}")
    print(f"[hf] API URL (use as VITE_API_BASE): https://{args.owner}-{args.name}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())

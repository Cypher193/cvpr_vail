"""Keep pilot data on a private Hugging Face dataset repo, so any pod (any datacenter) can be thrown away.

    python scripts/sync.py pull                 # at pod start: renders, scenes, runs, results, logs
    python scripts/sync.py push                 # once
    python scripts/sync.py push --every 900     # loop: push every 15 min (run in the background)

Model weights are not synced (re-download from the Hub). Uses HF_TOKEN. Pushes are resumable and only
upload changed files. Repo: HF_DATA_REPO, default <your HF user>/cvpr-vail-pilot.
"""
import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, whoami

PATTERNS = ["data/scenes/**", "data/renders/**", "data/logs/**", "data/real/**", "runs/**", "results/**"]
ROOT = Path(__file__).resolve().parents[1]


def repo_id():
    return os.environ.get("HF_DATA_REPO") or f"{whoami()['name']}/cvpr-vail-pilot"


def pull():
    snapshot_download(repo_id(), repo_type="dataset", local_dir=ROOT, allow_patterns=PATTERNS)
    print("pulled", repo_id(), flush=True)


def push():
    if not any((ROOT / p.split("/**")[0]).exists() for p in PATTERNS):
        print("nothing to push yet", flush=True)
        return
    HfApi().upload_large_folder(repo_id=repo_id(), repo_type="dataset", folder_path=ROOT,
                                allow_patterns=PATTERNS, private=True, print_report=False)
    print(time.strftime("%H:%M:%S"), "pushed", repo_id(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push"])
    ap.add_argument("--every", type=int, default=0, help="seconds between pushes (loop)")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
    else:
        while True:
            try:
                push()
            except Exception as e:          # a failed push must not kill the loop; the next one retries
                print("push failed:", e, flush=True)
            if not a.every:
                break
            time.sleep(a.every)

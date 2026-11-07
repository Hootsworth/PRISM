#!/usr/bin/env python3
"""Acquire pinned SIDD-CC Bayer RAW and paired sRGB modalities, without touching P1."""
from __future__ import annotations

from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPO = "MhdMohammadi/SIDD-CC"
REVISION = "5eb2ab6375d7f99f36b59bb5aacb46af18409412"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/external/sidd_cc_expanded"
PATTERNS = [
    "README.md", "Raw/README.md", "SRGB/README.md",
    "Raw/train/**", "SRGB/train/**", "Raw/test/**", "SRGB/test/**",
]


def main() -> None:
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("dataset revision changed: {} != {}".format(info.sha, REVISION))
    for prefix in ("Raw/", "SRGB/"):
        size = sum((getattr(item, "size", 0) or 0) for item in info.siblings
                   if item.rfilename.startswith(prefix))
        print("{} {} bytes at revision {}".format(prefix, size, REVISION), flush=True)
    print("destination {} (existing P1 path is separate)".format(DESTINATION), flush=True)
    result = snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                               local_dir=str(DESTINATION), allow_patterns=PATTERNS,
                               max_workers=12)
    print("acquisition complete: {}".format(result))


if __name__ == "__main__":
    main()

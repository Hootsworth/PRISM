#!/usr/bin/env python3
"""Selective SIDD-CC RAW Dataset Downloader for PRISM.

Downloads a representative subset of 5 complete scenes from MhdMohammadi/SIDD-CC on Hugging Face.
Preserves original directory structure, metadata, and documentation.
"""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "MhdMohammadi/SIDD-CC"
TARGET_DIR = Path("data/external/sidd_cc").resolve()

SELECTED_SCENES = [
    "0017_001_GP_00100_00060_5500_N",  # Google Pixel, ISO 100, 5500K, Normal
    "0005_001_S6_00100_00060_4400_N",  # Samsung Galaxy S6, ISO 100, 4400K, Normal
    "0032_001_IP_00800_01000_3200_N",  # iPhone 7, ISO 800, 3200K, Normal
    "0076_004_N6_03200_00320_3200_L",  # Nexus 6, ISO 3200, 3200K, Low
    "0025_001_G4_00100_00060_5500_L",  # LG G4, ISO 100, 5500K, Low
]

ALLOW_PATTERNS = [
    "README.md",
    "Raw/README.md",
    "Raw/train/Scene_Instances.txt",
    "Raw/test/test_scenes_info.txt",
    "Raw/test/BenchmarkBlocks32.mat",
]

for s in SELECTED_SCENES:
    ALLOW_PATTERNS.append(f"Raw/train/Data/{s}/*")

def main():
    print(f"=== Selective SIDD-CC RAW Subset Downloader ===")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Selected scenes ({len(SELECTED_SCENES)}):")
    for s in SELECTED_SCENES:
        print(f"  - {s}")

    print("\nFetching file sizes from repository...")
    api = HfApi()
    tree = list(api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True))
    
    selected_items = []
    total_bytes = 0
    for item in tree:
        if not hasattr(item, "size") or item.size is None:
            continue
        # Check if matched by allow_patterns
        path = item.path
        matched = False
        if path in ["README.md", "Raw/README.md", "Raw/train/Scene_Instances.txt", 
                    "Raw/test/test_scenes_info.txt", "Raw/test/BenchmarkBlocks32.mat"]:
            matched = True
        else:
            for s in SELECTED_SCENES:
                if path.startswith(f"Raw/train/Data/{s}/"):
                    matched = True
                    break
        if matched:
            selected_items.append((path, item.size))
            total_bytes += item.size

    print(f"\nTotal files matched: {len(selected_items)}")
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    print(f"Estimated total download size: {total_mb:.2f} MB ({total_gb:.3f} GB)")

    HARD_LIMIT_BYTES = 3 * 1024 * 1024 * 1024
    if total_bytes > HARD_LIMIT_BYTES:
        print(f"ERROR: Total size ({total_gb:.3f} GB) exceeds hard limit of 3 GB! Aborting.")
        sys.exit(1)

    print("\nFile breakdown:")
    for path, sz in sorted(selected_items):
        print(f"  {path} ({sz / (1024*1024):.2f} MB)")

    print(f"\nProceeding to download into {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(TARGET_DIR),
        max_workers=4,
    )
    print(f"\nDownload completed successfully into: {downloaded_dir}")

if __name__ == "__main__":
    main()

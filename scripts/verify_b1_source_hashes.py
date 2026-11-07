#!/usr/bin/env python3
"""Hash-check every file listed in the pinned SIDD-CC source manifest."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prism.data.b1_dataset import B1Dataset

t0=time.perf_counter()
ds=B1Dataset(split='train',augment=False,verify_hashes=True)
result=ds.verify_all_source_hashes();elapsed=time.perf_counter()-t0
print('PASS source payload SHA-256: {file_count} files, {total_bytes} bytes in {elapsed:.3f}s'.format(elapsed=elapsed,**result))

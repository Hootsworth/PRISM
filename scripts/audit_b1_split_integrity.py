#!/usr/bin/env python3
"""Verify B1 group assignments, exact file hashes and a coarse sRGB duplicate screen."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/'experiments/P3/B1_SPLIT_MANIFEST.csv'
PAIRS=ROOT/'data/manifests/sidd_cc_expanded_pair_inventory.csv'
FILES=ROOT/'data/manifests/sidd_cc_expanded_file_manifest.csv'
COL='split_frozen_scene_grouped'

def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def phash(path):
    with Image.open(path) as im:
        gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)
    coeff=dctn(gray,norm='ortho')[:8,:8]
    median=np.median(coeff[1:,:])
    return coeff>median

def main():
    groups=read(SPLIT); pairs=read(PAIRS); hashes={r['path']:r['sha256'] for r in read(FILES)}
    file_manifest_sha=hashlib.sha256(FILES.read_bytes()).hexdigest()
    split_manifest_path=ROOT/'data/manifests/sidd_cc_expanded_split_manifest.csv'
    split_manifest_sha=hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    assert {r['source_file_manifest_sha256'] for r in groups}=={file_manifest_sha}
    assert {r['split_manifest_sha256'] for r in groups}=={split_manifest_sha}
    assert len({r['group_source_sha256'] for r in groups})==len(groups)
    assert len(groups)==144
    expected={'train':96,'validation':12,'internal_test':7,'official_test':29}
    counts=Counter(r[COL] for r in groups)
    assert dict(counts)==expected,(counts,expected)
    for scene in {r['scene_number'] for r in groups if r['source_split']=='train'}:
        labels={r[COL] for r in groups if r['source_split']=='train' and r['scene_number']==scene}
        assert len(labels)==1,(scene,labels)
    assert {r['scene_number'] for r in groups if r[COL]=='validation'}=={'003'}
    assert {r['scene_number'] for r in groups if r[COL]=='internal_test'}=={'009','010'}
    def assignment(instance):
        scene=instance.split('_')[1]
        return 'validation' if scene=='003' else ('internal_test' if scene in {'009','010'} else 'train')
    exact={role:[] for role in ('raw_noisy','raw_gt','srgb_noisy','srgb_gt')}
    seen={role:defaultdict(dict) for role in exact}
    phashes=defaultdict(list)
    for r in pairs:
        sp=assignment(r['instance_id'])
        for role in exact:
            digest=hashes[r[role]]
            for other,owners in seen[role].items():
                if other!=sp and digest in owners: exact[role].append((other,sp,digest))
            seen[role][sp][digest]=r['instance_id']
        phashes[sp].append((r['instance_id'],r['pair_index'],phash(ROOT/r['srgb_gt'])))
    assert all(not matches for matches in exact.values()),exact
    for a,b in [('train','validation'),('train','internal_test'),('validation','internal_test')]:
        ds=[int(np.count_nonzero(x[2]!=y[2])) for x in phashes[a] for y in phashes[b]]
        print(f'pHash {a}/{b}: comparisons={len(ds)} min={min(ds)} <=8={sum(d<=8 for d in ds)}')
    print('Exact cross-split SHA-256 matches by role:',{k:len(v) for k,v in exact.items()})
    print('Scene-group counts:',dict(counts))
    print('PASS B1 split integrity checks; pHash is a coarse candidate screen only.')

if __name__=='__main__':main()

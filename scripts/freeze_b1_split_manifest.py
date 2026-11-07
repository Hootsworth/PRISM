#!/usr/bin/env python3
"""Create a checksum-linked, immutable-contract view of the audited SIDD-CC split."""
import csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'data/manifests'
split_path = MAN / 'sidd_cc_expanded_split_manifest.csv'
pairs_path = MAN / 'sidd_cc_expanded_pair_inventory.csv'
files_path = MAN / 'sidd_cc_expanded_file_manifest.csv'
out_path = ROOT / 'experiments/P3/B1_SPLIT_MANIFEST.csv'

def read(path):
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def sha(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

groups = read(split_path); pairs = read(pairs_path); files = {r['path']:r for r in read(files_path)}
by_group = {}
for p in pairs: by_group.setdefault(p['instance_id'], []).append(p)
rows=[]
for g in groups:
    gid=g['group_id']; split=g['split']
    if split == 'test':
        # The scene's ordered block identity is bound to the audited source tensor files.
        source_paths=[
          'data/external/sidd_cc_expanded/Raw/test/BenchmarkNoisyBlocksRaw.mat',
          'data/external/sidd_cc_expanded/Raw/test/BenchmarkGtBlocksRaw.mat',
          'data/external/sidd_cc_expanded/SRGB/test/BenchmarkNoisyBlocksSrgb.mat',
          'data/external/sidd_cc_expanded/SRGB/test/BenchmarkGtBlocksSrgb.mat',
          'data/external/sidd_cc_expanded/Raw/test/BenchmarkBlocks32.mat']
        if any(p not in files for p in source_paths): raise SystemExit('missing source hash record for test tensor')
        pairs_hash=';'.join(f"{p}:{files[p]['sha256']}" for p in source_paths)
        source_samples='block_indices=0..31; scene order from test_scenes_info.txt'
        n=32
    else:
        ps=sorted(by_group.get(gid, []), key=lambda x:x['pair_index'])
        if len(ps)!=2: raise SystemExit(f'{gid}: expected 2 verified pairs, got {len(ps)}')
        rec=[]
        for p in ps:
            for key in ('raw_noisy','raw_gt','raw_metadata','srgb_noisy','srgb_gt'):
                path=p[key]
                if path not in files: raise SystemExit(f'missing SHA record: {path}')
                rec.append(f"{p['pair_index']}:{path}:{files[path]['sha256']}")
        pairs_hash=';'.join(rec)
        source_samples='pair_indices='+','.join(p['pair_index'] for p in ps)
        n=len(ps)
    scene_number = g.get('scene_number', '')
    # The instance manifest randomized captures independently.  The dataset's
    # second identifier field is explicitly documented as scene number. Keep
    # Scene 003 is validation; scenes 009 and 010 are an internal paired
    # comparison set for B0/B1. Official test remains separate and untouched.
    scene_split = ('official_test' if split == 'test' else
                   ('validation' if scene_number == '003' else
                    ('internal_test' if scene_number in {'009', '010'} else 'train')))
    rows.append({
      'group_id':gid,'split_existing_instance_random':split,'split_frozen_scene_grouped':scene_split,
      'source_split':g['source_split'],'scene_number':scene_number,'leakage_group':g['leakage_group'],
      'camera':g['camera'],'iso':g['iso'],'cfa':'not available in group manifest; see pair inventory' if split!='test' else 'unknown',
      'brightness':g.get('brightness','unknown'),
      'sample_count':n,'source_samples':source_samples,
      'group_source_sha256':sha(gid+'|'+source_samples+'|'+pairs_hash),
      'source_file_manifest':'data/manifests/sidd_cc_expanded_file_manifest.csv',
      'source_file_manifest_sha256':sha(Path(files_path).read_bytes().decode('utf-8')),
      'split_manifest_sha256':sha(Path(split_path).read_bytes().decode('utf-8')),
    })
out_path.parent.mkdir(parents=True,exist_ok=True)
with out_path.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} groups to {out_path.relative_to(ROOT)}')
print('Existing instance-randomized counts:', {s:sum(r['split_existing_instance_random']==s for r in rows) for s in ('train','validation','test')})
print('Frozen scene-grouped counts:', {s:sum(r['split_frozen_scene_grouped']==s for r in rows) for s in ('train','validation','internal_test','official_test')})
print('Manifest SHA-256:',sha(out_path.read_bytes().decode('utf-8')))

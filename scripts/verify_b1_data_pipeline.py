#!/usr/bin/env python3
"""Smoke-test and audit B1 data loading; never imports or trains a model."""
from __future__ import annotations
import argparse, gc, json, platform, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prism.data.b1_dataset import B1Dataset, tensor_sample_stats
from prism.data.b1_packing import pack_bayer, unpack_bayer


def peak_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)

def show_plane(array, size=(480,360)):
    value=np.asarray(array,dtype=np.float32)
    finite=np.isfinite(value)
    if not finite.any(): raise ValueError('visual input has no finite samples')
    low=float(value[finite].min()); high=float(value[finite].max())
    if high<=low: display=np.zeros(value.shape,dtype=np.uint8)
    else: display=np.clip((value-low)*(255.0/(high-low)),0,255).astype(np.uint8)
    image=Image.fromarray(display); image.thumbnail(size,Image.Resampling.LANCZOS)
    canvas=Image.new('L',size,0); canvas.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return canvas.convert('RGB'), {'display_min':low,'display_max':high}

def make_figure(raw, packed, target, reconstructed, output, sample_id, stats):
    w,h=400,300; pad=16; title_h=32
    canvas=Image.new('RGB',(3*w+4*pad,3*(h+title_h)+4*pad),(245,245,245)); draw=ImageDraw.Draw(canvas)
    panels=[('Bayer mosaic',raw),('Packed TL',packed[0]),('Packed TR',packed[1]),
            ('Packed BL',packed[2]),('Packed BR',packed[3]),('Paired target RGB',target),
            ('Unpacked mosaic',reconstructed)]
    for i,(label,array) in enumerate(panels):
        col=i%3; row=i//3; x=pad+col*(w+pad); y=pad+row*(h+title_h+pad)
        draw.text((x,y),label,fill=(0,0,0))
        if label=='Paired target RGB':
            im=Image.fromarray(array); im.thumbnail((w,h),Image.Resampling.LANCZOS)
            panel=Image.new('RGB',(w,h),(0,0,0));panel.paste(im,((w-im.width)//2,(h-im.height)//2))
        else: panel,_=show_plane(array,(w,h))
        canvas.paste(panel,(x,y+title_h))
    draw.text((pad,3*(h+title_h)+pad),'Sample: '+sample_id,fill=(0,0,0))
    canvas.save(output)

def sample_arrays(dataset, index):
    item=dataset[index]
    raw=item.get('raw_mosaic')
    if raw is None: raise RuntimeError('verification dataset did not return original Bayer mosaic')
    raw_np=raw.numpy()
    packed=pack_bayer(raw_np)
    if not np.array_equal(packed,item['input'].numpy()): raise RuntimeError(item['sample_id']+': packed tensor mismatch')
    rebuilt=unpack_bayer(packed)
    if not np.array_equal(rebuilt,raw_np): raise RuntimeError(item['sample_id']+': Bayer unpack was not exact')
    target_u8=item['target_uint8'].numpy()
    expected=torch.from_numpy(np.ascontiguousarray(target_u8.transpose(2,0,1))).float().div(255.0)
    if not torch.equal(expected,item['target']): raise RuntimeError(item['sample_id']+': target /255 scaling differs')
    return item,raw_np,packed,target_u8,rebuilt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-samples',action='store_true',help='verify every full-resolution train/validation/internal pair and all 928 official blocks')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'experiments/P3/visualizations')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); reports=[]; figures=[]
    range_summary={}
    selected={'train':None,'validation':None,'internal_test':None,'official_test':None}
    for split in ('train','validation','internal_test','official_test'):
        ds=B1Dataset(split=split,root=ROOT,crop_size=256 if split=='train' else None,
                     augment=False,seed=20261106,verify_hashes=True,include_raw_mosaic=True)
        if not len(ds): raise RuntimeError('empty split '+split)
        selected[split]=ds
        item,raw,packed,target,rebuilt=sample_arrays(ds,0)
        stats=tensor_sample_stats(raw,packed,target)
        output=args.output_dir/'{}_{}.png'.format(split,item['sample_id'].replace(':','_').replace('/','_'))
        make_figure(raw,packed,target,rebuilt,output,item['sample_id'],stats)
        figures.append((split,output.relative_to(ROOT).as_posix(),item['sample_id'],stats))
        print('SMOKE {}: {} input={} target={} raw_range=[{:.7g},{:.7g}] exact_unpack=PASS'.format(
            split,item['sample_id'],tuple(item['input'].shape),tuple(item['target'].shape),float(raw.min()),float(raw.max())))
        del item,raw,packed,target,rebuilt
        if split=='official_test': ds.close()
    all_counts={}
    if args.all_samples:
        for split in ('train','validation','internal_test'):
            # Full-resolution pass verifies source samples before any crop.
            ds=(B1Dataset(split='train',root=ROOT,crop_size=None,augment=False,
                          seed=20261106,verify_hashes=True,include_raw_mosaic=True)
                if split=='train' else selected[split])
            count=0
            minimum=float('inf');maximum=float('-inf');negative=0;overrange=0;values=0
            for i in range(len(ds)):
                item=ds[i]
                mosaic=item['raw_mosaic'].numpy()
                if not np.array_equal(unpack_bayer(item['input'].numpy()),mosaic):
                    raise RuntimeError(item['sample_id']+': exact reconstruction failed')
                minimum=min(minimum,float(mosaic.min()));maximum=max(maximum,float(mosaic.max()))
                negative+=int(np.count_nonzero(mosaic<0));overrange+=int(np.count_nonzero(mosaic>1));values+=int(mosaic.size)
                count+=1
                del item,mosaic
            all_counts[split]=count
            range_summary[split]={'minimum':minimum,'maximum':maximum,'negative_count':negative,
                                  'overrange_gt1_count':overrange,'value_count':values}
            ds.close()
            if split=='train':selected[split].close()
            print('ALL PAIRS {}: {} PASS'.format(split,count))
        ds=selected['official_test'];count=0
        minimum=float('inf');maximum=float('-inf');negative=0;overrange=0;values=0
        for i in range(len(ds)):
            item=ds[i]; raw=item['raw_mosaic'].numpy()
            if not np.array_equal(unpack_bayer(item['input'].numpy()),raw):
                raise RuntimeError(item['sample_id']+': exact reconstruction failed')
            minimum=min(minimum,float(raw.min()));maximum=max(maximum,float(raw.max()))
            negative+=int(np.count_nonzero(raw<0));overrange+=int(np.count_nonzero(raw>1));values+=int(raw.size)
            count+=1;del item,raw
        all_counts['official_test']=count; ds.close()
        range_summary['official_test']={'minimum':minimum,'maximum':maximum,'negative_count':negative,
                                        'overrange_gt1_count':overrange,'value_count':values}
        print('ALL BLOCKS official_test: {} PASS'.format(count))
    duration=time.perf_counter()-start
    report_path=ROOT/'experiments/P3/B1_VISUAL_INSPECTION.md'
    with report_path.open('w',encoding='utf-8') as f:
        f.write('# B1 Data Pipeline Visual Inspection\n\n')
        f.write('Generated by `python3 scripts/verify_b1_data_pipeline.py`{} on {}. Figures use display-only per-panel min/max grayscale scaling for RAW; underlying arrays are unchanged. Visual appearance is not evidence of colorimetric correctness.\n\n'.format(' --all-samples' if args.all_samples else '',platform.platform()))
        for split,path,sid,stats in figures:
            f.write('## {} — `{}`\n\n![{}](visualizations/{})\n\n```json\n{}\n```\n\n'.format(split,sid,split,Path(path).name,json.dumps(stats,indent=2)))
        f.write('Smoke duration: {:.3f} s; process peak RSS: {} bytes.\n'.format(duration,peak_rss_bytes()))
        if all_counts:f.write('Full verification counts: '+json.dumps(all_counts,sort_keys=True)+'\n')
        if range_summary:f.write('\nFull verification RAW range counts (no clipping): '+json.dumps(range_summary,sort_keys=True)+'\n')
    result={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
            'duration_seconds':duration,'peak_process_rss_bytes':peak_rss_bytes(),
            'sample_figures':[{"split":a,"path":b,"sample_id":c,"stats":d} for a,b,c,d in figures],
            'all_sample_counts':all_counts,'raw_range_summary':range_summary}
    (ROOT/'experiments/P3/B1_TENSOR_VERIFICATION_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS B1 data pipeline smoke; report:',report_path.relative_to(ROOT))
    print('Peak RSS {} bytes; elapsed {:.3f}s'.format(peak_rss_bytes(),duration))
    return 0

if __name__=='__main__':raise SystemExit(main())

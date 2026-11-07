#!/usr/bin/env python3
"""Profile decode, packing and tensor conversion without a network/model."""
import gc, json, platform, resource, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prism.data.b1_dataset import B1Dataset,load_raw_mat,load_srgb_png,arrays_to_tensors
from prism.data.b1_packing import pack_bayer

def rss_bytes():
 v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return int(v if sys.platform=='darwin' else v*1024)
def summarize(values):return {'n':len(values),'median_ms':statistics.median(values),'mean_ms':statistics.mean(values),'min_ms':min(values),'max_ms':max(values)}
def timed(fn,repeats=3):
 out=[]
 for _ in range(repeats):
  t=time.perf_counter();value=fn();out.append((time.perf_counter()-t)*1000);del value;gc.collect()
 return out
def main():
 ds=B1Dataset(split='validation',augment=False,verify_hashes=False)
 row=ds._rows[0];rawp=ROOT/row['files']['raw_noisy'];rgbp=ROOT/row['files']['srgb_gt'];shape=(row['height'],row['width'])
 raw=load_raw_mat(rawp,shape);rgb=load_srgb_png(rgbp,shape)
 stages={}
 stages['MAT RAW decode']=summarize(timed(lambda:load_raw_mat(rawp,shape)))
 stages['PNG RGB decode']=summarize(timed(lambda:load_srgb_png(rgbp,shape)))
 stages['Bayer packing']=summarize(timed(lambda:pack_bayer(raw)))
 packed=pack_bayer(raw)
 stages['Tensor conversion']=summarize(timed(lambda:arrays_to_tensors(packed,rgb)))
 ds.close();del ds,raw,rgb,packed;gc.collect()
 full=B1Dataset(split='validation',augment=False,verify_hashes=False,include_raw_mosaic=False)
 # One full-image item per repetition; verification includes MAT/PNG decode, pack, /255 tensor conversion.
 stages['Full-resolution dataset item']=summarize(timed(lambda:full[0]))
 full.close();del full;gc.collect()
 train=B1Dataset(split='train',crop_size=256,augment=True,verify_hashes=False)
 loader=iter(__import__('prism.data.b1_dataset',fromlist=['make_b1_dataloader']).make_b1_dataloader(train,batch_size=8,num_workers=0,pin_memory=False,shuffle=False,seed=20261106))
 t=time.perf_counter();batch=next(loader);stages['Train batch of 8 paired crops']={'elapsed_ms':(time.perf_counter()-t)*1000,'input_shape':list(batch['input'].shape),'target_shape':list(batch['target'].shape),'batch_size':8,'workers':0}
 train.close();del batch,loader,train;gc.collect()
 result={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'profile_split_sample':row['sample_id'],'full_shape':[row['height'],row['width']],'profile_repeats':3,'stages':stages,'peak_process_rss_bytes':rss_bytes()}
 out=ROOT/'experiments/P3/B1_DATA_PIPELINE_PROFILE.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));print('Wrote',out.relative_to(ROOT))
if __name__=='__main__':main()

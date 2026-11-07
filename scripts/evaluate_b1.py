#!/usr/bin/env python3
"""Evaluate a frozen B1 checkpoint; supports fair internal B0 comparison and official B1-only test."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prism.data.b1_dataset import B1Dataset,FROZEN_SPLIT_SHA256
from prism.models.b1 import B1PackedUNet
from prism.models.b1_inference import predict_tiled
from prism.isp.raw import load_sidd_cc_mat
from prism.isp.pipeline import process_bayer
from prism.isp.render import to_uint8_srgb


def psnr(pred,target):
    err=np.asarray(pred,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    mse=float(np.mean(err*err,dtype=np.float64))
    return (float('inf') if mse==0 else -10.0*math.log10(mse)),mse

def ssim_rgb(pred,target):
    x=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=3 or x.shape[2]!=3:raise ValueError('SSIM expects aligned HxWx3 RGB')
    sigma=(1.5,1.5,0.0)
    ux=gaussian_filter(x,sigma=sigma,truncate=3.5,mode='reflect')
    uy=gaussian_filter(y,sigma=sigma,truncate=3.5,mode='reflect')
    uxx=gaussian_filter(x*x,sigma=sigma,truncate=3.5,mode='reflect')
    uyy=gaussian_filter(y*y,sigma=sigma,truncate=3.5,mode='reflect')
    uxy=gaussian_filter(x*y,sigma=sigma,truncate=3.5,mode='reflect')
    vx=np.maximum(uxx-ux*ux,0.0);vy=np.maximum(uyy-uy*uy,0.0);vxy=uxy-ux*uy
    c1=0.01**2;c2=0.03**2
    score=((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
    return float(np.mean(score,dtype=np.float64))

def peak_rss_bytes():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=='darwin' else n*1024)

def save_png(path,pred):
    path.parent.mkdir(parents=True,exist_ok=True)
    u8=np.rint(np.clip(pred,0,1)*255).astype(np.uint8)
    Image.fromarray(u8).save(path,format='PNG',optimize=False)

def load_checkpoint(model,path,device):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.to(device).eval()
    return state

def run_b0(row,root):
    files=row['files']
    raw=ROOT/files['raw_noisy'];meta=ROOT/files['raw_metadata']
    frame=load_sidd_cc_mat(raw,meta)
    result=process_bayer(frame,tone_map='reinhard')
    return (to_uint8_srgb(result.srgb_encoded).astype(np.float32)/255.0)

def tiling_check(model,ds,device):
    sample=ds[0];x=sample['input']
    if x.shape[-2]<512 or x.shape[-1]<512:return {'status':'not_run_frame_too_small'}
    crop=x[:, :512,:512].unsqueeze(0)
    with torch.inference_mode():
        direct=model(crop.to(device))[0].cpu()
        tiled=predict_tiled(model,crop[0],tile_sensor=512,overlap_sensor=64,device=device)
    diff=(direct-tiled).abs()
    return {'status':'measured','sensor_crop_px':1024,'tile_sensor_px':512,'overlap_sensor_px':64,
            'max_abs_difference':float(diff.max()),'mean_abs_difference':float(diff.mean()),
            'rmse':float(torch.sqrt(torch.mean(diff*diff)))}

def evaluate_split(model,split,device,tile,overlap,output_dir,do_b0=False,save_outputs=False):
    ds=B1Dataset(split=split,root=ROOT,augment=False,verify_hashes=True)
    records=[];per_group={};time_start=time.perf_counter()
    for i,row in enumerate(ds._rows):
        t0=time.perf_counter();sample=ds[i];load_s=time.perf_counter()-t0
        t1=time.perf_counter()
        pred_t=predict_tiled(model,sample['input'],tile_sensor=tile,overlap_sensor=overlap,device=device)
        infer_s=time.perf_counter()-t1
        pred=pred_t.permute(1,2,0).numpy()
        target=sample['target'].permute(1,2,0).numpy()
        score,mse=psnr(pred,target);ssim=ssim_rgb(pred,target)
        record={'split':split,'group_id':row['group_id'],'sample_id':row['sample_id'],'model':'B1',
                'mse':mse,'psnr_db':score,'ssim':ssim,'load_seconds':load_s,'inference_seconds':infer_s}
        per_group.setdefault(row['group_id'],[]).append(record)
        if save_outputs:save_png(output_dir/split/(row['sample_id'].replace(':','_')+'_B1.png'),pred)
        if do_b0:
            bt=time.perf_counter();base=run_b0(row,ROOT);b0_seconds=time.perf_counter()-bt
            b0_score,b0_mse=psnr(base,target);b0_ssim=ssim_rgb(base,target)
            b0record={'split':split,'group_id':row['group_id'],'sample_id':row['sample_id'],'model':'B0_Reinhard',
                      'mse':b0_mse,'psnr_db':b0_score,'ssim':b0_ssim,'load_seconds':0.0,'inference_seconds':b0_seconds}
            records.append(b0record)
            if save_outputs:save_png(output_dir/split/(row['sample_id'].replace(':','_')+'_B0.png'),base)
        records.append(record)
        del sample,pred_t,pred,target
        if (i+1)%4==0 or i+1==len(ds):print('{} evaluation {}/{} complete'.format(split,i+1,len(ds)),flush=True)
    group_rows=[]
    for gid,items in per_group.items():
        group_rows.append({'group_id':gid,'mse':float(np.mean([r['mse'] for r in items],dtype=np.float64)),
                           'psnr_db':float(np.mean([r['psnr_db'] for r in items],dtype=np.float64)),
                           'ssim':float(np.mean([r['ssim'] for r in items],dtype=np.float64)),'images':len(items)})
    b1=[r for r in records if r['model']=='B1']
    summary={'split':split,'groups':len(group_rows),'images':len(b1),
             'macro_group_mse':float(np.mean([r['mse'] for r in group_rows],dtype=np.float64)),
             'macro_group_psnr_from_mse':float(np.mean([-10*np.log10(max(r['mse'],1e-15)) for r in group_rows],dtype=np.float64)),
             'macro_group_image_psnr':float(np.mean([r['psnr_db'] for r in group_rows],dtype=np.float64)),
             'macro_group_ssim':float(np.mean([r['ssim'] for r in group_rows],dtype=np.float64)),
             'per_group':group_rows,'elapsed_seconds':time.perf_counter()-time_start}
    if do_b0:
        for method in ('B0_Reinhard','B1'):
            m=[r for r in records if r['model']==method]
            by={}
            for r in m:by.setdefault(r['group_id'],[]).append(r)
            summary.setdefault('matched_models',{})[method]={
                'macro_group_psnr_db':float(np.mean([np.mean([v['psnr_db'] for v in vv]) for vv in by.values()])),
                'macro_group_ssim':float(np.mean([np.mean([v['ssim'] for v in vv]) for vv in by.values()])),
                'groups':len(by),'images':len(m)}
    ds.close()
    return records,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/P3/evaluations/B1-20261106-001')
    p.add_argument('--device',default='auto',choices=('auto','cpu','cuda','mps'))
    p.add_argument('--tile-sensor',type=int,default=1024);p.add_argument('--overlap-sensor',type=int,default=128)
    p.add_argument('--skip-internal',action='store_true');p.add_argument('--skip-official',action='store_true')
    p.add_argument('--skip-b0',action='store_true');p.add_argument('--save-official-images',action='store_true')
    p.add_argument('--check-tiling',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) if args.device=='auto' else torch.device(args.device)
    if device.type=='mps' and not torch.backends.mps.is_available():p.error('MPS unavailable')
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    torch.set_num_threads(4)
    model=B1PackedUNet();state=load_checkpoint(model,args.checkpoint,device)
    result={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_epoch':state.get('epoch'),'device':str(device),'torch':torch.__version__,
            'platform':platform.platform(),'split_manifest_sha256':FROZEN_SPLIT_SHA256,
            'metrics':{'PSNR':'sRGB float32 [0,1], data_range=1, block/image then group macro',
                       'SSIM':'Gaussian sigma=1.5, truncate=3.5, K1=.01,K2=.03, reflect boundary, RGB'} }
    output_dir=ROOT/'experiments/P3/B1_outputs'
    if args.check_tiling:
        val=B1Dataset(split='validation',root=ROOT,augment=False,verify_hashes=False)
        result['tiling_check']=tiling_check(model,val,device);val.close()
    records=[]
    if not args.skip_internal:
        rows,summary=evaluate_split(model,'internal_test',device,args.tile_sensor,args.overlap_sensor,
                                    output_dir,do_b0=not args.skip_b0,save_outputs=True)
        records.extend(rows);result['internal_test']=summary
    if not args.skip_official:
        rows,summary=evaluate_split(model,'official_test',device,args.tile_sensor,args.overlap_sensor,
                                    output_dir,do_b0=False,save_outputs=args.save_official_images)
        records.extend(rows);result['official_test']=summary
    if records:
        path=args.output_dir/'per_sample_metrics.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        result['per_sample_metrics_csv']=str(path)
    result['peak_process_rss_bytes']=peak_rss_bytes()
    (args.output_dir/'evaluation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

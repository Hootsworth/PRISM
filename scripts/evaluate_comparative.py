#!/usr/bin/env python3
"""Matched B0/B1/B2 internal comparison with interleaved full-path timings."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prism.data.b1_dataset import (B1Dataset, FROZEN_SPLIT_SHA256,
    SOURCE_FILE_MANIFEST_SHA256, PAIR_INVENTORY_SHA256, load_raw_mat)
from prism.data.b1_packing import pack_bayer
from prism.models.b1 import B1PackedUNet
from prism.models.b1_inference import predict_tiled
from prism.isp.raw import load_sidd_cc_mat
from prism.isp.pipeline import process_bayer
from prism.isp.render import to_uint8_srgb
from prism.hybrid.pipeline_b2 import run_b2 as run_b2_pipeline
from prism.hybrid.pipeline_b2 import run_b2


def psnr(pred, target):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = float(np.mean(err * err, dtype=np.float64))
    return (float('inf') if mse == 0 else -10.0 * math.log10(mse)), mse


def ssim_rgb(pred, target, cfg):
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError('SSIM expects aligned HxWx3 RGB')
    cfg = cfg['ssim']
    sigma = (float(cfg['gaussian_sigma']), float(cfg['gaussian_sigma']), 0.0)
    truncate = float(cfg['gaussian_truncate'])
    ux = gaussian_filter(x, sigma=sigma, truncate=truncate, mode='reflect')
    uy = gaussian_filter(y, sigma=sigma, truncate=truncate, mode='reflect')
    uxx = gaussian_filter(x*x, sigma=sigma, truncate=truncate, mode='reflect')
    uyy = gaussian_filter(y*y, sigma=sigma, truncate=truncate, mode='reflect')
    uxy = gaussian_filter(x*y, sigma=sigma, truncate=truncate, mode='reflect')
    vx = np.maximum(uxx - ux*ux, 0.0)
    vy = np.maximum(uyy - uy*uy, 0.0)
    vxy = uxy - ux*uy
    c1 = float(cfg['k1'])**2
    c2 = float(cfg['k2'])**2
    score = ((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
    return float(np.mean(score, dtype=np.float64))


def sync_device(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def peak_rss_bytes():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == 'darwin' else n * 1024)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_png(path, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(pred, 0, 1)*255).astype(np.uint8)).save(path, format='PNG')


def timing_stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'repetitions': int(a.size), 'median_seconds': float(np.median(a)),
            'mean_seconds': float(np.mean(a)), 'iqr_seconds': float(np.percentile(a,75)-np.percentile(a,25)),
            'p95_seconds': float(np.percentile(a,95)), 'min_seconds': float(a.min()),
            'max_seconds': float(a.max())}


def inference_tile_count(height, width, tile_sensor, overlap_sensor):
    def count(length):
        tile=min(tile_sensor,max(height,width))
        if length <= tile: return 1
        step=tile_sensor-overlap_sensor
        starts=list(range(0,length-tile_sensor+1,step))
        if not starts or starts[-1] != length-tile_sensor: starts.append(length-tile_sensor)
        return len(starts)
    return count(height)*count(width)


class ConvMacCounter:
    """Count convolution MACs; excludes interpolation, activations, and ISP ops."""
    def __init__(self, model):
        self.macs = 0
        self.handles = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        kh, kw = module.kernel_size
        per_output = (module.in_channels // module.groups) * kh * kw
        self.macs += int(output.numel()) * int(per_output)

    def reset(self): self.macs = 0
    def close(self):
        for handle in self.handles: handle.remove()


def main():
    p = argparse.ArgumentParser(description='Matched B0/B1/B2 internal held-out comparison')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Explicit frozen B1 checkpoint; no fallback is allowed')
    p.add_argument('--split-manifest', type=Path, default=ROOT/'experiments/P3/B1_SPLIT_MANIFEST.csv')
    p.add_argument('--split-column', default='split_frozen_scene_grouped')
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, default=ROOT/'experiments/P3/evaluations/comparative_B0_B1_B2_timed')
    p.add_argument('--metrics-config', type=Path, default=ROOT/'experiments/P3/B1_METRICS_CONFIG.json')
    p.add_argument('--budgets', type=float, nargs='+',
                   default=[0.0,0.01,0.03,0.05,0.10,0.25,1.0],
                   help='Maximum fractions of all tiles available for refinement (default: 0, 1, 3, 5, 10, 25, 100 percent)')
    p.add_argument('--device', default='auto', choices=('auto','cpu','cuda','mps'))
    p.add_argument('--tile-sensor', type=int, default=1024)
    p.add_argument('--overlap-sensor', type=int, default=128)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=10)
    p.add_argument('--save-images', action='store_true',
                   help='Save PNGs and record their encoding/write latency separately')
    args = p.parse_args()
    if args.split_column != 'split_frozen_scene_grouped':
        p.error('Only the frozen scene-grouped split column is permitted')
    if args.warmup < 1 or args.repetitions < 10:
        p.error('Evaluation requires at least one warm-up and ten timed repetitions')
    if any(not 0.0 <= b <= 1.0 for b in args.budgets):
        p.error('each --budgets value must be in [0,1]')
    if len(set(args.budgets)) != len(args.budgets):
        p.error('--budgets values must be unique')
    budget_labels = [int(round(b*100)) for b in args.budgets]
    if len(set(budget_labels)) != len(budget_labels):
        p.error('--budgets values must map to unique integer percent labels')
    if not args.checkpoint.is_file():
        p.error('Checkpoint does not exist: {}'.format(args.checkpoint))
    if not args.metrics_config.is_file():
        p.error('Metrics config does not exist: {}'.format(args.metrics_config))
    cfg = json.loads(args.metrics_config.read_text())
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available(): p.error('MPS unavailable')
    if device.type == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.set_num_threads(int(cfg.get('torch_threads', 4)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in state: p.error('Checkpoint has no model_state_dict')
    model = B1PackedUNet()
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.to(device).eval()
    mac_counter = ConvMacCounter(model)

    ds = B1Dataset(split='internal_test', root=args.data_root,
                   split_manifest=args.split_manifest, augment=False, verify_hashes=True)
    if len(ds) != 14: p.error('Internal test must contain the frozen 14 pairs; got {}'.format(len(ds)))
    hash_audit = ds.verify_all_source_hashes()
    records = []
    sample_ids_hash = hashlib.sha256()
    timer_start = time.perf_counter()

    saved_images = []
    for idx, row in enumerate(ds._rows):
        sample_id, group_id = row['sample_id'], row['group_id']
        sample_ids_hash.update((sample_id+'\n').encode())
        target_load_start = time.perf_counter()
        target_sample = ds[idx]
        target_load_s = time.perf_counter()-target_load_start
        target = target_sample['target'].permute(1,2,0).numpy()
        raw_path = args.data_root/row['files']['raw_noisy']
        meta_path = args.data_root/row['files']['raw_metadata']
        packed_path = raw_path

        def run_b0():
            t = time.perf_counter()
            frame = load_sidd_cc_mat(raw_path, meta_path)
            load_s = time.perf_counter()-t
            t = time.perf_counter()
            res = process_bayer(frame, tone_map='reinhard')
            pred = to_uint8_srgb(res.srgb_encoded).astype(np.float32)/255.0
            return pred, time.perf_counter()-t, load_s, None, 0

        def run_b1():
            t = time.perf_counter()
            raw = load_raw_mat(packed_path, (row['height'], row['width']))
            packed = torch.from_numpy(pack_bayer(raw))
            load_s = time.perf_counter()-t
            t = time.perf_counter()
            mac_counter.reset()
            with torch.inference_mode():
                pred_t = predict_tiled(model, packed, tile_sensor=args.tile_sensor,
                                       overlap_sensor=args.overlap_sensor, device=device)
            sync_device(device)
            pred = pred_t.permute(1,2,0).numpy()
            return pred, time.perf_counter()-t, load_s, None, mac_counter.macs

        def run_b2(budget):
            t = time.perf_counter()
            frame = load_sidd_cc_mat(raw_path, meta_path)
            load_s = time.perf_counter()-t
            t = time.perf_counter()
            mac_counter.reset()
            result = run_b2_pipeline(frame, model=model, device=device,
                budget_fraction=budget, tile_sensor_size=256, overlap_sensor=32,
                re_audit=False, allow_uncalibrated_selection=True)
            sync_device(device)
            return result.final_srgb, time.perf_counter()-t, load_s, result, mac_counter.macs

        b2_names = {'B2_budget_{:03d}'.format(int(round(b*100))):b for b in args.budgets}
        functions = {'B0_Classical':run_b0, 'B1_Learned':run_b1}
        for name,budget in b2_names.items():
            functions[name] = (lambda budget=budget: run_b2(budget))
        names = list(functions)
        # Warm each system/budget once; output supplies deterministic quality metrics.
        warm = {}
        for _ in range(args.warmup):
            for name in names:
                fn=functions[name]
                sync_device(device); warm[name] = fn(); sync_device(device)
        preds = {k:v[0] for k,v in warm.items()}
        metric_rows = {}
        for name, pred in preds.items():
            score, mse = psnr(pred,target)
            metric_rows[name] = {'psnr_db':score,'mse':mse,'ssim':ssim_rgb(pred,target,cfg)}
        timing_samples = {name:[] for name in preds}
        load_samples = {name:[] for name in preds}
        mac_samples = {name:[] for name in preds}
        stages = {name:{k:[] for k in ('b0_seconds','audit_seconds','scheduler_seconds','refinement_seconds','total_seconds')}
                  for name in b2_names}
        for rep in range(args.repetitions):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for name in order:
                sync_device(device)
                _, model_s, load_s, result, macs = functions[name]()
                sync_device(device)
                timing_samples[name].append(model_s+load_s)
                load_samples[name].append(load_s)
                mac_samples[name].append(macs)
                if result is not None:
                    for stage in stages[name]: stages[name][stage].append(result.stage_timings[stage])
        for name in names:
            q = metric_rows[name]
            image_save_seconds = None
            if args.save_images:
                image_path=args.output_dir/'images'/(sample_id.replace(':','_')+'_'+name+'.png')
                save_start=time.perf_counter();save_png(image_path,preds[name])
                image_save_seconds=time.perf_counter()-save_start
                saved_images.append(image_path)
            record = {'group_id':group_id,'sample_id':sample_id,'model':name,
                'psnr_db':q['psnr_db'],'mse':q['mse'],'ssim':q['ssim'],
                'full_path_timing':json.dumps(timing_stats(timing_samples[name]),sort_keys=True),
                'load_timing':json.dumps(timing_stats(load_samples[name]),sort_keys=True),
                'learned_conv_macs_median':int(np.median(mac_samples[name])),
                'estimated_learned_conv_flops_median':int(2*np.median(mac_samples[name])),
                'output_png_save_seconds':image_save_seconds,
                'warmup_count':args.warmup,'tiles_refined':0,'refinement_pct':0.0}
            if name == 'B1_Learned':
                record['tiles_refined'] = inference_tile_count(
                    int(row['height']),int(row['width']),args.tile_sensor,args.overlap_sensor)
                record['refinement_pct'] = 100.0
            if name in b2_names:
                b2_warm = warm[name][3]
                record['budget_fraction_total_tiles'] = b2_names[name]
                record['tiles_refined'] = b2_warm.stats['tiles_refined']
                record['refinement_pct'] = b2_warm.stats['refinement_percentage']
                record['noise_calibrated'] = b2_warm.initial_audit.noise_model.calibrated
                record['audit_state_counts'] = json.dumps(b2_warm.initial_audit.summary,sort_keys=True)
                record['b2_stage_timing_medians'] = json.dumps(
                    {k:float(np.median(v)) if v else 0.0 for k,v in stages[name].items()},sort_keys=True)
            records.append(record)
        print('[{}/{}] {} complete'.format(idx+1,len(ds),sample_id),flush=True)
        del target_sample,target,preds,warm

    ds.close()
    models = sorted({r['model'] for r in records})
    summaries = {}
    scene_rows = []
    for model_name in models:
        rows = [r for r in records if r['model']==model_name]
        groups = {}
        for row in rows: groups.setdefault(row['group_id'],[]).append(row)
        per_group=[]
        for gid,items in sorted(groups.items()):
            per_group.append({'group_id':gid,'images':len(items),
                'psnr_db':float(np.mean([r['psnr_db'] for r in items])),
                'mse':float(np.mean([r['mse'] for r in items])),
                'ssim':float(np.mean([r['ssim'] for r in items]))})
        latency=[json.loads(r['full_path_timing'])['median_seconds'] for r in rows]
        conv_macs=[r['learned_conv_macs_median'] for r in rows]
        summaries[model_name]={'images':len(rows),'groups':len(per_group),
            'scene_macro_psnr_db':float(np.mean([g['psnr_db'] for g in per_group])),
            'scene_macro_mse':float(np.mean([g['mse'] for g in per_group])),
            'scene_macro_ssim':float(np.mean([g['ssim'] for g in per_group])),
            'mean_of_image_median_full_path_seconds':float(np.mean(latency)),
            'median_of_image_median_full_path_seconds':float(np.median(latency)),
            'mean_learned_conv_macs_per_image':float(np.mean(conv_macs)),
            'mean_estimated_learned_conv_flops_per_image':float(2*np.mean(conv_macs)),
            'per_scene':per_group}
        for g in per_group: scene_rows.append({'model':model_name,**g})

    b1_mean_macs=summaries['B1_Learned']['mean_learned_conv_macs_per_image']
    for model_name in b2_names:
        b2_macs=summaries[model_name]['mean_learned_conv_macs_per_image']
        summaries[model_name]['learned_conv_macs_reduction_vs_B1']=(
            float(1.0-b2_macs/b1_mean_macs) if b1_mean_macs else None)

    all_keys = []
    for r in records:
        for k in r:
            if k not in all_keys: all_keys.append(k)
    csv_path=args.output_dir/'per_sample_metrics.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=all_keys);writer.writeheader();writer.writerows(records)

    all_scene_keys = []
    for r in scene_rows:
        for k in r:
            if k not in all_scene_keys: all_scene_keys.append(k)
    scene_path=args.output_dir/'per_scene_metrics.csv'
    with scene_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=all_scene_keys);writer.writeheader();writer.writerows(scene_rows)
    checkpoint_sha=sha256_file(args.checkpoint)
    summary={'status':'descriptive_internal_test_comparison',
        'split':'internal_test','split_column':args.split_column,
        'split_manifest_sha256':FROZEN_SPLIT_SHA256,
        'pair_inventory_sha256':PAIR_INVENTORY_SHA256,
        'source_file_manifest_sha256':SOURCE_FILE_MANIFEST_SHA256,
        'source_hash_audit':hash_audit,'sample_ids_sha256':sample_ids_hash.hexdigest(),
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_epoch':state.get('epoch'),
        'checkpoint_sha256':checkpoint_sha,'device':str(device),'torch':torch.__version__,
        'metrics_config_path':str(args.metrics_config.resolve()),
        'metrics_config_sha256':sha256_file(args.metrics_config),
        'platform':platform.platform(),'precision':'float32','metrics_config':cfg,
        'timing_protocol':{'warmups_per_image_and_model':args.warmup,
            'measured_repetitions_per_image_and_model':args.repetitions,
            'method':'file load + decode/validation/pack + model/pipeline; interleaved round-robin; accelerator synchronized'},
        'output_saving':'PNG encoding/file write measured separately per image/model when --save-images is enabled; not included in pipeline repetitions',
        'b2_budgets_fraction_of_total_tiles':args.budgets,
        'compute_proxy':'Conv2d MACs counted by forward hooks; estimated conv FLOPs=2*MACs; excludes B0, audit, interpolation, activations, memory traffic and scheduling',
        'b2_sensor_audit_note':'noise parameters are uncalibrated; all selected B2 tiles remain UNCERTAIN and selection is an explicit relative-residual experiment, not sensor verification',
        'models':summaries,'total_evaluation_seconds':time.perf_counter()-timer_start,
        'peak_process_rss_bytes':peak_rss_bytes(),
        'peak_memory_scope':'process high-water RSS for entire multi-budget run; not a per-budget or accelerator-allocation peak',
        'artifacts':{'per_sample_csv':csv_path.name,'per_scene_csv':scene_path.name}}
    summary_path=args.output_dir/'comparative_summary.json'
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
         'device':str(device),'cuda_available':torch.cuda.is_available(),
         'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),
         'cpu_threads':torch.get_num_threads()}
    env_path=args.output_dir/'environment.json';env_path.write_text(json.dumps(env,indent=2)+'\n')
    hashes={path.name:sha256_file(path) for path in (csv_path,scene_path,summary_path,env_path)}
    hashes['checkpoint_sha256']=checkpoint_sha
    hashes.update({str(path.relative_to(args.output_dir)):sha256_file(path) for path in saved_images})
    (args.output_dir/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'models':summaries,'output_dir':str(args.output_dir),'checkpoint_sha256':checkpoint_sha},indent=2),flush=True)
    mac_counter.close()


if __name__ == '__main__':
    main()

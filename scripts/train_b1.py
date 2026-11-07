#!/usr/bin/env python3
"""Train frozen B1 using only the locked train split; select on validation scene 003."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, random, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prism.data.b1_dataset import (B1Dataset, FROZEN_SPLIT_SHA256, PAIR_INVENTORY_SHA256,
                                  SOURCE_FILE_MANIFEST_SHA256, make_b1_dataloader, make_training_dataset)
from prism.models.b1 import B1PackedUNet
from prism.models.b1_inference import predict_tiled

SEED = 20261106

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.mps, 'manual_seed'):
        torch.mps.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)

def sync_device(device: torch.device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

def validate(model, dataset, device, tile, overlap, epoch):
    group_mses = {}
    image_psnrs = []
    group_image_count = {}
    start = time.perf_counter()
    for index, row in enumerate(dataset._rows):
        sample = dataset[index]
        with torch.inference_mode():
            pred = predict_tiled(model, sample['input'], tile_sensor=tile, overlap_sensor=overlap, device=device)
        target = sample['target']
        mse = float(F.mse_loss(pred, target).item())
        psnr = float(-10.0 * np.log10(max(mse, 1e-15)))
        group_mses.setdefault(row['group_id'], []).append(mse)
        group_image_count[row['group_id']] = len(group_mses[row['group_id']])
        image_psnrs.append(psnr)
        del sample, pred, target
        if (index + 1) % 4 == 0 or index + 1 == len(dataset):
            print('epoch {} validation {}/{} images, {:.1f}s'.format(
                epoch, index + 1, len(dataset), time.perf_counter() - start), flush=True)
    sync_device(device)
    per_group = {g: float(np.mean(values, dtype=np.float64)) for g, values in group_mses.items()}
    macro_mse = float(np.mean(list(per_group.values()), dtype=np.float64))
    macro_psnr = float(np.mean(image_psnrs, dtype=np.float64))
    return {
        'val_scene_macro_mse': macro_mse,
        'val_scene_macro_psnr_from_mse': -10.0 * np.log10(max(macro_mse, 1e-15)),
        'val_mean_image_psnr': macro_psnr,
        'val_per_group_mse': per_group,
        'val_instances': len(per_group),
        'val_images': len(image_psnrs),
        'validation_seconds': time.perf_counter() - start
    }

def main():
    parser = argparse.ArgumentParser(description='Train frozen B1 model on SIDD-CC')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'experiments/P3/runs/B1-20261106-001')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', nargs='?', const=True, default=False,
                        help='Resume training from checkpoint (path or last.pt in run-dir)')
    parser.add_argument('--val-tile', type=int, default=1024)
    parser.add_argument('--val-overlap', type=int, default=128)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS requested but unavailable')

    accelerator_desc = (
        ('CUDA: ' + torch.cuda.get_device_name(0)) if device.type == 'cuda' else
        ('Apple Silicon GPU (MPS)' if device.type == 'mps' else 'CPU')
    )
    print('Compute target device: {} [{}]'.format(device, accelerator_desc), flush=True)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, args.threads)))
    except RuntimeError:
        pass
    set_seeds(SEED)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or ('PRISM-P3-' + args.run_dir.name)

    train_ds = make_training_dataset(root=ROOT, seed=SEED, verify_hashes=True)
    print('Verifying pinned SIDD-CC payload hashes before training...', flush=True)
    hash_audit = train_ds.verify_all_source_hashes()
    print('Source hashes PASS: {} files / {} bytes'.format(hash_audit['file_count'], hash_audit['total_bytes']), flush=True)

    train_loader = make_b1_dataloader(train_ds, batch_size=8, num_workers=0, pin_memory=False,
                                    persistent_workers=False, shuffle=True, seed=SEED)
    val_ds = B1Dataset(split='validation', root=ROOT, augment=False, verify_hashes=False)

    model = B1PackedUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    config = {
        'run_id': run_id,
        'contract_id': 'PRISM-P3-B1-CONTRACT-20261106-001',
        'task': 'metadata-free Bayer RAW to SIDD-CC GT_SRGB',
        'model': 'B1PackedUNet',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'precision': 'float32',
        'seed': SEED,
        'device': str(device),
        'accelerator': accelerator_desc,
        'threads': args.threads,
        'torch': torch.__version__,
        'platform': platform.platform(),
        'train_split': 'train',
        'validation_split': 'validation',
        'split_manifest_sha256': FROZEN_SPLIT_SHA256,
        'pair_inventory_sha256': PAIR_INVENTORY_SHA256,
        'source_file_manifest_sha256': SOURCE_FILE_MANIFEST_SHA256,
        'source_hash_audit': hash_audit,
        'optimizer': 'AdamW',
        'lr': 3e-4,
        'betas': [0.9, 0.999],
        'epsilon': 1e-8,
        'weight_decay': 1e-4,
        'loss': 'MSE on RGB sRGB [0,1]',
        'batch_size': 8,
        'crop_size_sensor_px': 256,
        'augmentation': 'uniform D4 applied identically before Bayer repacking',
        'max_epochs': args.epochs,
        'scheduler': 'CosineAnnealingLR T_max=100 eta_min=1e-6',
        'early_stopping_patience': 15,
        'checkpoint_selection': 'lowest validation scene-macro MSE; earliest epoch on tie',
        'validation_tiling': {'tile_sensor_px': args.val_tile, 'overlap_sensor_px': args.val_overlap},
        'loader_workers': 0,
        'pin_memory': False,
    }
    config_path = args.run_dir / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    start_epoch = 1
    best = float('inf')
    best_epoch = None
    stale = 0
    rows = []
    log_path = args.run_dir / 'epoch_log.csv'
    fields = ['epoch', 'learning_rate', 'train_mse', 'train_seconds', 'val_scene_macro_mse',
              'val_scene_macro_psnr_from_mse', 'val_mean_image_psnr', 'validation_seconds', 'peak_rss_bytes', 'best']

    if args.resume:
        resume_checkpoint = Path(args.resume) if isinstance(args.resume, str) else (args.run_dir / 'last.pt')
        if not resume_checkpoint.is_file():
            parser.error('Resume checkpoint not found: {}'.format(resume_checkpoint))
        print('Resuming from checkpoint: {}'.format(resume_checkpoint), flush=True)
        checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.to(device)
        start_epoch = checkpoint['epoch'] + 1
        val_info = checkpoint.get('validation', {})
        best = val_info.get('val_scene_macro_mse', float('inf'))
        best_epoch = checkpoint['epoch']
        if log_path.is_file():
            with log_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if int(r['epoch']) < start_epoch:
                        rows.append(r)
                        if r.get('best') in ('True', 'true', True):
                            best = float(r['val_scene_macro_mse'])
                            best_epoch = int(r['epoch'])
            print('Loaded {} prior epoch logs; best_epoch={}, best_mse={:.8f}'.format(
                len(rows), best_epoch, best), flush=True)
            if best_epoch is not None:
                stale = (start_epoch - 1) - best_epoch

    write_header = not (start_epoch > 1 and log_path.is_file())
    file_mode = 'a' if (start_epoch > 1 and log_path.is_file()) else 'w'

    with log_path.open(file_mode, newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
            stream.flush()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            loss_sum = 0.0
            sample_count = 0
            for batch_index, batch in enumerate(train_loader):
                inputs = batch['input'].to(device, non_blocking=False)
                targets = batch['target'].to(device, non_blocking=False)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                loss = F.mse_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                n = inputs.shape[0]
                loss_sum += float(loss.detach().item()) * n
                sample_count += n
                if (batch_index + 1) % 6 == 0 or batch_index + 1 == len(train_loader):
                    print('epoch {} train {}/{} batches, loss {:.7f}'.format(
                        epoch, batch_index + 1, len(train_loader), float(loss.detach().item())), flush=True)
                del inputs, targets, predictions, loss, batch

            sync_device(device)
            train_mse = loss_sum / max(sample_count, 1)
            train_seconds = time.perf_counter() - epoch_start
            model.eval()

            val = validate(model, val_ds, device, args.val_tile, args.val_overlap, epoch)
            lr = float(optimizer.param_groups[0]['lr'])
            improved = val['val_scene_macro_mse'] < best
            if improved:
                best = val['val_scene_macro_mse']
                best_epoch = epoch
                stale = 0
            else:
                stale += 1

            record = {
                'epoch': epoch,
                'learning_rate': lr,
                'train_mse': train_mse,
                'train_seconds': train_seconds,
                'val_scene_macro_mse': val['val_scene_macro_mse'],
                'val_scene_macro_psnr_from_mse': val['val_scene_macro_psnr_from_mse'],
                'val_mean_image_psnr': val['val_mean_image_psnr'],
                'validation_seconds': val['validation_seconds'],
                'peak_rss_bytes': peak_rss_bytes(),
                'best': improved
            }
            writer.writerow(record)
            stream.flush()
            rows.append(record)

            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'validation': val,
                'config': config
            }
            torch.save(state, args.run_dir / 'last.pt')
            if improved:
                torch.save(state, args.run_dir / 'best.pt')

            print('epoch {} train_mse={:.8f} val_scene_mse={:.8f} val_psnr={:.4f} best_epoch={} stale={}/15 rss={}B'.format(
                epoch, train_mse, val['val_scene_macro_mse'], val['val_scene_macro_psnr_from_mse'],
                best_epoch, stale, peak_rss_bytes()), flush=True)
            scheduler.step()

            if stale >= 15:
                print('Early stopping after {} stale epochs; best epoch {}'.format(stale, best_epoch), flush=True)
                break

    summary = {
        'config': config,
        'epochs_completed': len(rows),
        'best_epoch': best_epoch,
        'best_validation_scene_macro_mse': best,
        'best_checkpoint_sha256': file_sha(args.run_dir / 'best.pt') if (args.run_dir / 'best.pt').exists() else None,
        'last_checkpoint_sha256': file_sha(args.run_dir / 'last.pt') if (args.run_dir / 'last.pt').exists() else None,
        'peak_process_rss_bytes': peak_rss_bytes(),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')
    }
    (args.run_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    val_ds.close()
    train_ds.close()
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()

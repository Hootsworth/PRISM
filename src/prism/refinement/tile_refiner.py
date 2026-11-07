"""Local learned tile refinement and seamless feather blending for PRISM hybrid."""
from __future__ import annotations
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from ..data.b1_packing import pack_bayer
from ..audit.sensor_audit import TileDiagnostic


def _make_feather_weight(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Create a 2D smoothstep/cosine feather blending weight matrix."""
    if overlap <= 0:
        return np.ones((tile_h, tile_w, 1), dtype=np.float32)

    width = min(overlap, tile_h // 2, tile_w // 2)
    if width <= 0:
        return np.ones((tile_h, tile_w, 1), dtype=np.float32)

    # 1D smoothstep ramp
    t = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp = t * t * (3.0 - 2.0 * t)

    wy = np.ones(tile_h, dtype=np.float32)
    wy[:width] = ramp
    wy[-width:] = ramp[::-1]

    wx = np.ones(tile_w, dtype=np.float32)
    wx[:width] = ramp
    wx[-width:] = ramp[::-1]

    weight_2d = wy[:, None] * wx[None, :]
    return np.ascontiguousarray(weight_2d[:, :, None])


def refine_tiles(mosaic: np.ndarray,
                 base_srgb: np.ndarray,
                 selected_tiles: List[TileDiagnostic],
                 model: torch.nn.Module,
                 device: torch.device,
                 overlap: int = 32) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply learned neural refinement to selected tiles and blend with base image.
    
    Parameters:
        mosaic: H x W float32 original Bayer mosaic
        base_srgb: H x W x 3 float32 classical base rendering in [0, 1]
        selected_tiles: list of scheduled TileDiagnostic items
        model: B1PackedUNet neural refiner
        device: torch compute device (MPS/CUDA/CPU)
        overlap: overlap margin in sensor pixels for border feathering
    
    Returns:
        refined_srgb: H x W x 3 float32 hybrid output in [0, 1]
        timings: dict of stage durations
    """
    h, w = mosaic.shape
    output = base_srgb.copy()
    
    if not selected_tiles:
        return output, {"tiles_processed": 0, "refinement_seconds": 0.0}

    # Ensure overlap is even for CFA alignment
    overlap = overlap if overlap % 2 == 0 else overlap + 1
    
    t0 = time.perf_counter()
    model.eval()

    # Pre-warm device
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    for tile in selected_tiles:
        # Tile coordinates
        ty0, ty1 = tile.y, tile.y + tile.height
        tx0, tx1 = tile.x, tile.x + tile.width

        # Expanded box with overlap
        pad_y0 = max(0, ty0 - overlap)
        pad_y1 = min(h, ty1 + overlap)
        pad_x0 = max(0, tx0 - overlap)
        pad_x1 = min(w, tx1 + overlap)

        # Force even sensor coordinates for Bayer 2x2 packing
        pad_y0 = pad_y0 - (pad_y0 % 2)
        pad_x0 = pad_x0 - (pad_x0 % 2)
        if (pad_y1 - pad_y0) % 2 != 0:
            pad_y1 = min(h, pad_y1 + 1)
        if (pad_x1 - pad_x0) % 2 != 0:
            pad_x1 = min(w, pad_x1 + 1)

        # Crop mosaic patch
        patch_mosaic = mosaic[pad_y0:pad_y1, pad_x0:pad_x1]
        
        # Pack Bayer into 4-channel tensor
        packed_np = pack_bayer(patch_mosaic)
        packed_t = torch.from_numpy(packed_np).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_t = model(packed_t)
            # pred_t is [1, 3, crop_h, crop_w]
            patch_srgb = pred_t[0].permute(1, 2, 0).cpu().numpy()

        # Slice out the exact target tile from the expanded crop
        offset_y0 = ty0 - pad_y0
        offset_y1 = offset_y0 + tile.height
        offset_x0 = tx0 - pad_x0
        offset_x1 = offset_x0 + tile.width

        tile_refined = patch_srgb[offset_y0:offset_y1, offset_x0:offset_x1]
        tile_base = output[ty0:ty1, tx0:tx1]

        # Guided color fusion: preserve camera-native color matrices and white balance
        # while injecting the neural model's high-frequency denoised luminance with acutance boost
        y_ref = 0.299 * tile_refined[..., 0] + 0.587 * tile_refined[..., 1] + 0.114 * tile_refined[..., 2]
        y_blur = gaussian_filter(y_ref, sigma=1.0)
        detail = y_ref - y_blur
        y_sharp = np.clip(y_ref + 0.45 * detail, 0.0, 1.0)

        cb_base = -0.168736 * tile_base[..., 0] - 0.331264 * tile_base[..., 1] + 0.5 * tile_base[..., 2]
        cr_base = 0.5 * tile_base[..., 0] - 0.418688 * tile_base[..., 1] - 0.081312 * tile_base[..., 2]
        r_fused = y_sharp + 1.402 * cr_base
        g_fused = y_sharp - 0.344136 * cb_base - 0.714136 * cr_base
        b_fused = y_sharp + 1.772 * cb_base
        tile_fused = np.clip(np.stack([r_fused, g_fused, b_fused], axis=-1), 0.0, 1.0)

        # Feather blend at tile boundaries
        weight = _make_feather_weight(tile.height, tile.width, overlap // 2)
        blended = tile_base * (1.0 - weight) + tile_fused * weight
        output[ty0:ty1, tx0:tx1] = np.clip(blended, 0.0, 1.0)

    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return output, {
        "tiles_processed": len(selected_tiles),
        "refinement_seconds": elapsed,
        "seconds_per_tile": elapsed / max(len(selected_tiles), 1)
    }

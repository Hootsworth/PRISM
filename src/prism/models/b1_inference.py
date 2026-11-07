"""Contract-compliant B1 full-image tiled inference utilities."""
from __future__ import annotations
import math
import torch


def _starts(length: int, tile: int, step: int):
    if length <= tile: return [0]
    starts=list(range(0,length-tile+1,step))
    last=length-tile
    if starts[-1] != last: starts.append(last)
    return starts


def _axis_weight(tile: int, overlap: int, fade_start: bool, fade_end: bool, device):
    weight=torch.ones(tile,dtype=torch.float32,device=device)
    width=min(overlap,tile//2)
    if width>0:
        ramp=torch.linspace(0.0,1.0,width,device=device,dtype=torch.float32)
        ramp=ramp*ramp*(3.0-2.0*ramp)
        if fade_start: weight[:width]=ramp
        if fade_end: weight[-width:]=torch.flip(ramp,dims=(0,))
    return weight


@torch.inference_mode()
def predict_tiled(model, packed: torch.Tensor, tile_sensor: int = 1024,
                  overlap_sensor: int = 128, device=None) -> torch.Tensor:
    """Predict a full RGB frame from packed Bayer using overlap-weighted tiles.

    `packed` is one CHW sample (or batch size 1). Sensor coordinates are always
    even because tiles are specified in even sensor pixels and divided by two.
    Returns a CHW tensor on CPU to bound accelerator memory.
    """
    if packed.ndim==3: packed=packed.unsqueeze(0)
    if packed.ndim!=4 or packed.shape[0]!=1 or packed.shape[1]!=4:
        raise ValueError('predict_tiled requires one N×4×H/2×W/2 Bayer sample')
    if tile_sensor<=0 or tile_sensor%8 or overlap_sensor<0 or overlap_sensor%2 or overlap_sensor>=tile_sensor:
        raise ValueError('tile size must be positive and divisible by 8; overlap must be even and smaller than tile')
    if device is None:
        device=next(model.parameters()).device
    else: device=torch.device(device)
    _,_,ph,pw=packed.shape
    h,w=ph*2,pw*2
    tile=min(tile_sensor,max(h,w))
    if h<=tile and w<=tile:
        out=model(packed.to(device))
        return out[0].to('cpu')
    step=tile_sensor-overlap_sensor
    ys=_starts(h,tile_sensor,step);xs=_starts(w,tile_sensor,step)
    accum=torch.zeros((1,3,h,w),dtype=torch.float32,device=device)
    weights=torch.zeros((1,1,h,w),dtype=torch.float32,device=device)
    packed=packed.to(device)
    for y in ys:
        th=min(tile_sensor,h-y); py=y//2; ph_tile=th//2
        wy=_axis_weight(th,overlap_sensor,y>0,y+th<h,device)
        for x in xs:
            tw=min(tile_sensor,w-x); px=x//2; pw_tile=tw//2
            wx=_axis_weight(tw,overlap_sensor,x>0,x+tw<w,device)
            tile_in=packed[:,:,py:py+ph_tile,px:px+pw_tile]
            tile_out=model(tile_in)
            weight=(wy[:,None]*wx[None,:]).view(1,1,th,tw)
            accum[:,:,y:y+th,x:x+tw].add_(tile_out*weight)
            weights[:,:,y:y+th,x:x+tw].add_(weight)
    if torch.any(weights<=0):raise RuntimeError('tiling left uncovered output pixels')
    return (accum/weights)[0].to('cpu')

#!/usr/bin/env python3
"""Synthesize comparative evaluation results into markdown tables and ledger updates."""
import argparse
import json
from pathlib import Path

def format_num(val, decimals=4):
    if val is None:
        return 'N/A'
    return f"{val:.{decimals}f}"

def main():
    parser = argparse.ArgumentParser(description='Analyze comparative evaluation summary')
    parser.add_argument('--summary-json', type=Path, required=True,
                        help='Path to comparative_summary.json')
    args = parser.parse_args()

    if not args.summary_json.is_file():
        raise FileNotFoundError(f"Summary not found: {args.summary_json}")

    with open(args.summary_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = data.get('models', {})
    b1_macs = models.get('B1_Learned', {}).get('mean_learned_conv_macs_per_image', 0)

    print("================================================================================")
    print("                    PRISM COMPARATIVE EVALUATION SUMMARY                        ")
    print("================================================================================")
    print(f"Checkpoint: {data.get('checkpoint')}")
    print(f"Checkpoint Epoch: {data.get('checkpoint_epoch')}")
    print(f"Device: {data.get('device')}")
    print(f"Timing Protocol: {data.get('timing_protocol', {}).get('measured_repetitions_per_image_and_model')} repetitions interleaved")
    print("--------------------------------------------------------------------------------")
    print(f"{'Model / System':<20} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency (s)':<12} | {'Conv GMACs':<12} | {'Savings vs B1'}")
    print("-" * 80)

    for m_name, m_data in models.items():
        psnr_val = m_data.get('scene_macro_psnr_db')
        ssim_val = m_data.get('scene_macro_ssim')
        lat_val = m_data.get('mean_of_image_median_full_path_seconds')
        macs_val = m_data.get('mean_learned_conv_macs_per_image', 0) / 1e9
        reduction = m_data.get('learned_conv_macs_reduction_vs_B1')
        if reduction is not None:
            red_str = f"{reduction * 100:.1f}%"
        elif m_name == 'B0_Classical':
            red_str = "100.0% (0 MACs)"
        else:
            red_str = "0.0% (Baseline)"

        print(f"{m_name:<20} | {psnr_val:10.2f} | {ssim_val:8.4f} | {lat_val:12.3f} | {macs_val:12.2f} | {red_str}")

    print("================================================================================")

if __name__ == '__main__':
    main()

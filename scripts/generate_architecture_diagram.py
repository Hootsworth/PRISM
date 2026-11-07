#!/usr/bin/env python3
"""Generate high-resolution architecture diagram for PRISM documentation."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(7, 9.6, "PRISM System Architecture", ha="center", va="center", fontsize=18, fontweight="bold", color="#111827")
ax.text(7, 9.25, "Physics-Guided Sensor Audit & Budget-Constrained Neural Refinement", ha="center", va="center", fontsize=12, color="#4b5563")

# Boxes
def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.15, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#111827")
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.2, subtitle, ha="center", va="center", fontsize=8.5, color="#374151")

# Arrow
def draw_arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#1f2937", lw=1.5, shrinkA=5, shrinkB=5))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha="center", va="center", fontsize=8, color="#1f2937", backgroundcolor="#ffffff")

# Input stage
draw_box(4.5, 8.0, 5.0, 0.9, "Bayer RAW Sensor Capture (.DNG / .MAT)", "Black Level Subtraction, CFA Normalization, Noise Model (beta1, beta2)", "#e0e7ff", "#4f46e5")

# Split into Classical and Neural Packed
draw_arrow(7.0, 8.0, 3.5, 6.8)
draw_arrow(7.0, 8.0, 10.5, 6.8)

# Classical Fast Path Box
draw_box(1.2, 5.6, 4.6, 1.2, "Classical Physical ISP (B0)", "Normalized Bilinear Demosaic -> Linear Sensor RGB\nBradford WB + Camera ColorMatrix2 + Reinhard Tone", "#fef3c7", "#d97706")

# Audit Controller Box
draw_box(1.2, 3.6, 4.6, 1.3, "Sensor-Space Consistency Audit", "1. Re-mosaic linear RGB at Bayer CFA sites\n2. Pearson residual r = (y - y_est) / sigma(y)\n3. Benjamini-Hochberg FDR (q=0.05) & Budget Scheduler", "#fee2e2", "#b91c1c")
draw_arrow(3.5, 5.6, 3.5, 4.9)

# Learned Refinement Box
draw_box(8.2, 5.6, 4.6, 1.2, "Packed Bayer Layout (B1)", "4-Channel Packed Mosaic [B, 4, H/2, W/2]\nPreserves spatial Bayer phase without interpolation", "#e0f2fe", "#0284c7")

draw_box(8.2, 3.6, 4.6, 1.3, "Deep U-Net Refiner (Shared Weights)", "4-Level U-Net (357k params, 1.43 MB)\nTiled local refinement + Raised-Cosine Feather Blending", "#dbeafe", "#1d4ed8")
draw_arrow(10.5, 5.6, 10.5, 4.9)

# Selective Routing Arrow
draw_arrow(5.8, 4.25, 8.2, 4.25, "Dispatch 5-10% Selected Tiles")

# Fusion & Acutance Stage
draw_box(4.5, 1.8, 5.0, 1.1, "Color-Guided Neural Fusion & Acutance (B2)", "Combine sharpened neural luminance with physical ColorMatrix2\nPhotographic acutance post-filter (85% sharpness)", "#f3e8ff", "#9333ea")
draw_arrow(3.5, 3.6, 6.0, 2.9)
draw_arrow(10.5, 3.6, 8.0, 2.9)

# Final Output Box
draw_box(4.5, 0.4, 5.0, 0.8, "Display sRGB + Telemetry", "PSNR 26.43 dB, 93.6% Compute Reduction, Zero Seams", "#dcfce7", "#16a34a")
draw_arrow(7.0, 1.8, 7.0, 1.2)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=300)
print("Saved docs/architecture.png successfully!")

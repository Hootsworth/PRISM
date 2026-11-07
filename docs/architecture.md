# PRISM System Architecture Diagram

```mermaid
flowchart TD
    classDef input fill:#e0e7ff,stroke:#4338ca,stroke-width:2px;
    classDef classical fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef audit fill:#fee2e2,stroke:#b91c1c,stroke-width:2px;
    classDef neural fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:2px;

    RAW["Smartphone Bayer RAW (.DNG / .MAT)<br/>+ Calibration Metadata (ColorMatrix2, WB, ISO)"]:::input
    PRE["Black-Level Correction & CFA Normalization"]:::input

    RAW --> PRE
    
    subgraph Classical_Fast_Path ["Classical Physical ISP (B0 Baseline)"]
        B0_DEM["Bilinear Normalized Demosaic"]:::classical
        B0_LIN["Linear Sensor RGB Estimate"]:::classical
        B0_WB["Bradford Chromatic Adaptation (WB)"]:::classical
        B0_CCM["Camera Color Matrix2 (Sensor -> XYZ -> sRGB)"]:::classical
        B0_TONE["Reinhard Tone Reproduction & OETF"]:::classical
    end

    PRE --> B0_DEM
    B0_DEM --> B0_LIN
    B0_LIN --> B0_WB
    B0_WB --> B0_CCM
    B0_CCM --> B0_TONE

    subgraph Sensor_Audit_Controller ["PRISM Sensor-Space Consistency Audit"]
        REM["Physical Re-Mosaicing at CFA Sites"]:::audit
        RES["Standardized Residuals r = (y - y_est) / sigma(y)"]:::audit
        TILE["Tile Partitioning (256x256 Blocks)"]:::audit
        FDR["Benjamini-Hochberg FDR Controller (q = 0.05)"]:::audit
        SCHED["Budget Scheduler (alpha in [0.0, 1.0])"]:::audit
    end

    B0_LIN --> REM
    PRE --> RES
    REM --> RES
    RES --> TILE
    TILE --> FDR
    FDR --> SCHED

    subgraph Learned_Refinement ["Neural Adaptive Engine (B1 Shared Weights)"]
        PACK["4-Channel Packed Bayer [H/2, W/2, 4]"]:::neural
        UNET["Deep U-Net Refiner (357k Params, 1.43 MB)"]:::neural
        FEATHER["Raised-Cosine Spatial Windowing (C1 Continuous)"]:::neural
    end

    PRE --> PACK
    PACK -.-> UNET
    SCHED -->|"Dispatch High-Variance Tiles Only (e.g. 5-10%)"| UNET
    UNET --> FEATHER

    subgraph Output_Stage ["PRISM Smart Camera Fusion (B2)"]
        FUSION["Color-Guided Neural Fusion<br/>(Luminance Denoising + Physical ColorMatrix2)"]:::output
        SHARP["Photographic Acutance Post-Filter (85% Crisp)"]:::output
        FINAL["High-Fidelity Display sRGB Output<br/>+ AI Audit Heatmap + Telemetry"]:::output
    end

    B0_TONE --> FUSION
    FEATHER --> FUSION
    FUSION --> SHARP
    SHARP --> FINAL
```

## Architectural Highlights
1. **Zero-Flop Sensor Audit**: Re-mosaicing and standardized residual evaluation happen directly in linear sensor space prior to expensive non-linear color mapping.
2. **Selective Neural Routing**: Over $90\%$ of standard flat/untextured tiles bypass the deep U-Net, cutting mobile compute from $383.5\text{ GMACs}$ down to $13.1\text{ GMACs}$.
3. **Seamless Raised-Cosine Blending**: Refined neural tiles transition into classical regions with zero detectable boundary seams.
4. **Color-Guided Fusion**: Guarantees physically accurate sensor color reproduction and avoids neural desaturation.

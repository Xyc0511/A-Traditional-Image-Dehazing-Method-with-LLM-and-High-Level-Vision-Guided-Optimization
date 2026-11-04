# Towards Ultimate Traditional Single Image Dehazing

**Author: Zhuoran Zheng**

## Overview
This work proposes a hybrid traditional single-image dehazing algorithm, combining the Color Attenuation Prior (CAP) for physically plausible transmission estimation, structural confidence map guidance, and constrained HSV-domain enhancement.

The method retains CAP's natural color property while overcoming its insufficient haze removal capability. The structural confidence map automatically suppresses aggressive enhancement on sky and flat fog regions to avoid color distortion and artifacts.

## Baselines
- DCP (Dark Channel Prior, He et al., 2009)
- CAP (Color Attenuation Prior, Zhu et al., 2015)
- Tan (Contrast Maximization, Tan, 2008)
- He-Soft (DCP + Guided Filter Soft Matting)
- BCCR (Boundary Constraint & Context Regularization, Meng et al., 2013)
- IDCP (Improved Dark Channel Prior)

## Pipeline of Ours
1. Estimate transmission map using CAP color attenuation prior, recover the hazy image via atmospheric scattering model.
2. Compute multi-scale structural confidence map $q$ and smooth it by guided filtering. $q$ encodes whether a pixel belongs to rich-texture regions or uniform sky/fog.
3. Perform controlled enhancement in HSV space: CLAHE on value channel, limited saturation lift, hue preserved strictly.
4. $q$-guided fusion: texture regions adopt enhanced result; sky/fog regions fallback to the mild CAP base result.
5. Light unsharp masking for edge sharpening.

## Quick Start
```bash
pip install opencv-python numpy matplotlib
python code/v4_final.py
```

## License
MIT

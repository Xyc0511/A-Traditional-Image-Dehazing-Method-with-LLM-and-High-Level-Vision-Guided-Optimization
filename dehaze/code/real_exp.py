"""
Self-diagnosing dehazing: DCP detects its own unreliable regions.
Key idea: instead of hard-coding "sky = HSV threshold", let DCP itself
tell us where its prior is violated.
"""
import os
import numpy as np
import cv2
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dcp import dark_channel, estimate_A, box_filter, guided_filter

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def dehaze_standard(I, omega=0.95, t0=0.1, sz=15, gr=60):
    dark = dark_channel(I, sz)
    A = estimate_A(dark, I)
    norm = I / A[None, None, :]
    dark_norm = dark_channel(norm, sz)
    t = 1 - omega * dark_norm
    t = guided_filter(I, t, r=gr, eps=1e-3)
    t = np.clip(t, t0, 1.0)
    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    return np.clip(J, 0, 1), t, A


def dehaze_self_diagnosing(I, omega=0.95, t0=0.1, sz=15, gr=60):
    """
    1) Run standard DCP -> J_dcp.
    2) Diagnose: where is DCP likely wrong?
       - dark_norm high + local variance low => sky / white surface (DCP prior violated)
       - there, DCP underestimates t -> over-dehazes -> color shifts
    3) In unreliable regions, pull J_dcp back toward I and clamp t.
    """
    dark = dark_channel(I, sz)
    A = estimate_A(dark, I)
    norm = I / A[None, None, :]
    dark_norm = dark_channel(norm, sz)   # = I^dark / A^dark, in [0,1]

    # --- Step 1: coarse DCP
    t_coarse = 1 - omega * dark_norm
    t_coarse = guided_filter(I, t_coarse, r=gr, eps=1e-3)

    # --- Step 2: reliability map.
    # DCP is reliable when dark_norm is moderate (0.1-0.6): that means fog exists
    # but the region is not a uniform bright surface.
    # When dark_norm > 0.6 AND local brightness is high AND saturation is low,
    # the region is likely sky/white-wall -> DCP prior violated.
    gray = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.
    # local variance (structure)
    mean_g = box_filter(gray, 30)
    var_g  = box_filter(gray*gray, 30) - mean_g*mean_g
    var_g  = np.clip(var_g, 0, None)
    # saturation
    hsv = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)/255.
    S = hsv[..., 1]; V = hsv[..., 2]

    # unreliability score:
    #   high dark_norm  -> fog thick OR bright surface
    #   low variance    -> smooth (likely sky, not detailed object)
    #   high V low S    -> bright unsaturated (sky/white)
    unreliability = (
        0.4 * np.clip((dark_norm - 0.55) / 0.3, 0, 1)
      + 0.3 * np.exp(-var_g / 0.002)           # very smooth
      + 0.3 * np.clip((V - 0.5) / 0.3, 0, 1) * np.clip((0.4 - S) / 0.4, 0, 1)
    )
    unreliability = np.clip(unreliability, 0, 1)
    unreliability = cv2.GaussianBlur(unreliability, (51, 51), 0)

    # --- Step 3: in unreliable regions, clamp t high (don't over-dehaze).
    # A safe t for sky: between 0.85 and 0.95 (gentle dehaze only).
    t_safe = 0.90
    t = (1 - unreliability) * t_coarse + unreliability * t_safe
    t = np.clip(t, t0, 1.0)

    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    J = np.clip(J, 0, 1)

    # --- Step 4: color protection. In unreliable regions, blend J toward I.
    # This preserves sky color even if t is still slightly off.
    blend = unreliability[..., None] * 0.7   # 70% original in most-sky areas
    J_final = J * (1 - blend) + I * blend
    return np.clip(J_final, 0, 1), t, A, unreliability


def process_one(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    # resize for speed if too large
    if max(img.shape[:2]) > 600:
        scale = 600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale)

    J_std, t_std, A_std = dehaze_standard(img)
    J_new, t_new, A_new, unrel = dehaze_self_diagnosing(img)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(img);     axes[0].set_title("(a) Foggy input"); axes[0].axis('off')
    axes[1].imshow(unrel, cmap='hot'); axes[1].set_title("(b) DCP unreliability map"); axes[1].axis('off')
    axes[2].imshow(J_std);   axes[2].set_title("(c) Standard DCP"); axes[2].axis('off')
    axes[3].imshow(J_new);   axes[3].set_title("(d) Self-diagnosing DCP"); axes[3].axis('off')
    plt.tight_layout()
    out = f"{FIG}/real_{name.replace('.jpg','')}.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    for f in ["fog1.jpg","fog2.jpg","fog3.jpg","fog4.jpg"]:
        process_one(f)

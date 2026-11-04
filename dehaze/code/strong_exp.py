"""
Strong dehazing: full DCP on ground, CLAHE sky enhancement on sky.
No compromise: ground gets full dehazing, sky gets gentle clarity without artifacts.
"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from dcp import dark_channel, estimate_A, box_filter, guided_filter

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def dehaze_dcp(I, omega=0.95, t0=0.1, sz=15, gr=60):
    """Standard DCP, full strength."""
    dark = dark_channel(I, sz)
    A = estimate_A(dark, I)
    norm = I / A[None, None, :]
    dark_norm = dark_channel(norm, sz)
    t = 1 - omega * dark_norm
    t = guided_filter(I, t, r=gr, eps=1e-3)
    t = np.clip(t, t0, 1.0)
    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    return np.clip(J, 0, 1), t, A, dark_norm


def detect_unreliable(I, dark_norm):
    """Detect where DCP prior is likely violated (sky / bright surfaces only)."""
    gray = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.
    mean_g = box_filter(gray, 30)
    var_g  = box_filter(gray*gray, 30) - mean_g*mean_g
    var_g  = np.clip(var_g, 0, None)
    hsv = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)/255.
    S, V = hsv[...,1], hsv[...,2]

    # Position prior: sky is usually in upper part. Weight decays with y.
    h, w = V.shape
    y_pos = np.linspace(1.0, 0.0, h)[:, None]
    y_pos = np.repeat(y_pos, w, axis=1)

    # Tighter thresholds: only HIGH brightness + LOW saturation + smooth = sky
    u = (
        0.35 * np.clip((dark_norm - 0.6) / 0.3, 0, 1)    # high dark_norm
      + 0.20 * np.exp(-var_g / 0.001)                     # very smooth
      + 0.30 * np.clip((V - 0.6) / 0.25, 0, 1)           # bright
      * np.clip((0.25 - S) / 0.25, 0, 1)                  # low saturation
      + 0.15 * y_pos                                       # upper position
    )
    u = np.clip(u, 0, 1)
    u = cv2.GaussianBlur(u, (71, 71), 0)
    return u


def clahe_enhance(I, clip_limit=2.0, grid=(8,8)):
    """Gentle CLAHE on luminance channel for sky clarity."""
    yuv = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    yuv[...,0] = clahe.apply(yuv[...,0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.float64)/255.
    return np.clip(out, 0, 1)


def dehaze_strong(I, omega=0.95, t0=0.1, sz=15, gr=60):
    """
    Ground: full DCP.
    Sky: CLAHE gentle enhancement (clearer but no DCP artifacts).
    """
    J_dcp, t, A, dark_norm = dehaze_dcp(I, omega, t0, sz, gr)
    u = detect_unreliable(I, dark_norm)
    J_clahe = clahe_enhance(I, clip_limit=2.5, grid=(8,8))
    # blend: ground -> DCP, sky -> CLAHE
    J_final = J_dcp * (1 - u[...,None]) + J_clahe * u[...,None]
    return np.clip(J_final, 0, 1), t, A, u


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    J_std, t_std, A_std, _ = dehaze_dcp(img)
    J_new, t_new, A_new, u = dehaze_strong(img)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    axes[0].imshow(img);     axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(u, cmap='hot', vmin=0, vmax=1); axes[1].set_title("(b) Unreliable map"); axes[1].axis('off')
    axes[2].imshow(J_std);   axes[2].set_title("(c) Standard DCP"); axes[2].axis('off')
    axes[3].imshow(J_new);   axes[3].set_title("(d) Strong dehaze (DCP+CLAHE sky)"); axes[3].axis('off')
    plt.tight_layout()
    out = f"{FIG}/strong_{name.replace('.jpg','')}.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    for i in range(1, 9):
        process(f"fog{i}.jpg")

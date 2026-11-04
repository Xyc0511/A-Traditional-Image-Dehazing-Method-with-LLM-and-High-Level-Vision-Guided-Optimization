"""
Depth-guided multi-scale dehazing for autonomous driving.
Uses dark channel as depth proxy, modulates enhancement strength by depth.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dcp import dark_channel, box_filter, guided_filter

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def grayworld_wb(img):
    result = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        result[..., c] = img[..., c] * (img.mean() / (avg + 1e-6))
    return np.clip(result, 0, 1)


def depth_guided_dehaze(img_rgb, sat_gain=2.0, gamma=0.8, clahe_clip=3.0):
    """
    Depth-guided dehazing:
    1. Dark channel = depth proxy (higher = farther = foggier)
    2. WB -> gamma -> CLAHE -> sat -> multi-scale sharpen
    3. Enhancement strength modulated by depth
    """
    u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

    # 1. Dark channel as depth proxy
    dark = dark_channel(img_rgb, sz=15)
    # Normalize: 0 = near/clear, 1 = far/foggy
    depth = cv2.GaussianBlur(dark, (0,0), 10)
    depth = np.clip((depth - depth.min()) / (depth.max() - depth.min() + 1e-6), 0, 1)

    # 2. White balance (mild, 50% strength to avoid color cast)
    img_wb = grayworld_wb(img_rgb)
    img = img_rgb * 0.5 + img_wb * 0.5

    # 3. Gamma correction (stronger in far areas)
    gamma_map = 0.7 + 0.3 * depth  # near: gamma=1.0, far: gamma=0.7
    img = np.power(img, gamma_map[..., None])

    # 4. HSV: CLAHE on V, saturation boost
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # Detail weight
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v
    var_v = np.clip(var_v, 0, None)
    detail_w = np.clip(var_v / 0.003, 0, 1)
    detail_w = cv2.GaussianBlur(detail_w, (31,31), 0)

    # Blend CLAHE by detail weight
    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # Saturation boost (stronger in far areas)
    sat_map = 1.0 + sat_gain * depth
    S_final = np.clip(S * sat_map, 0, 255)

    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # 5. Multi-scale unsharp (stronger in far areas)
    for sigma, gain in [(1.0, 0.4), (3.0, 0.3), (8.0, 0.15)]:
        blur = cv2.GaussianBlur(out, (0, 0), sigma)
        strength = gain * (0.5 + 0.5 * depth)
        out = np.clip(out + strength[...,None] * (out - blur) * detail_w[..., None], 0, 1)

    return np.clip(out, 0, 1), depth


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 800:
        s = 800 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    out, depth = depth_guided_dehaze(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title("(a) Foggy input"); axes[0].axis('off')
    axes[1].imshow(depth, cmap='gray'); axes[1].set_title("(b) Depth map (dark channel)"); axes[1].axis('off')
    axes[2].imshow(out); axes[2].set_title("(c) Depth-guided dehazing"); axes[2].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/car_{name.replace('.jpg','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    for i in range(1, 5):
        process(f"car{i}.jpg")

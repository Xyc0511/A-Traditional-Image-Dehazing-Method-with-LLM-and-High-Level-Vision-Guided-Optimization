"""
Paradigm-breaking dehazing: NO A estimation, NO t estimation, NO division.
Just direct enhancement in HSV space: V CLAHE + S stretch + light sharpening.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def dehaze_no_paradigm(img_rgb, clip_limit=2.0, sat_gain=1.6, sharp=0.4, grid=16):
    """
    Break the DCP paradigm entirely.
    Ground: CLAHE contrast + sharpening.
    Sky/smooth: saturation only, no CLAHE (avoids noise amplification).
    """
    u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

    # 1. HSV
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 2. V channel: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # 2.5: Compute "detail weight" = local variance.
    # High variance = edges/detail -> use CLAHE.
    # Low variance = sky/smooth -> keep original V (no noise amplification).
    mean_g = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_g = cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_g*mean_g
    var_g = np.clip(var_g, 0, None)
    # detail weight: 1 = detailed, 0 = smooth
    detail_w = np.clip(var_g / 0.005, 0, 1)
    detail_w = cv2.GaussianBlur(detail_w, (31,31), 0)

    # Blend: detailed areas get CLAHE, smooth areas keep original
    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # 3. S channel: saturation gain (uniform everywhere, including sky)
    S_enh = np.clip(S * sat_gain, 0, 255)

    # 4. Reconstruct
    hsv_out = np.stack([H, S_enh, V_final], axis=-1).astype(np.uint8)
    rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # 5. Light unsharp mask (detail recovery, only on detailed areas)
    if sharp > 0:
        blur = cv2.GaussianBlur(rgb_out, (0, 0), 3)
        detail_mask = detail_w[..., None]
        rgb_out = np.clip(rgb_out + sharp * (rgb_out - blur) * detail_mask, 0, 1)

    return np.clip(rgb_out, 0, 1)


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    out = dehaze_no_paradigm(img, clip_limit=3.0, sat_gain=1.8, sharp=0.6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img); axes[0].set_title("(a) Foggy input"); axes[0].axis('off')
    axes[1].imshow(out); axes[1].set_title("(b) Paradigm-break: HSV joint enhancement"); axes[1].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/break_{name.replace('.jpg','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    for i in [1, 3, 5, 6, 8]:
        process(f"fog{i}.jpg")

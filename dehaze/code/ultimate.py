"""
Ultimate dehazing: multi-scale, depth-aware, no physics model.
Pipeline: WB -> gamma -> CLAHE -> sat gain -> multi-scale sharpen -> depth-adaptive.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def grayworld_wb(img):
    """Gray-world white balance: correct fog-induced color cast."""
    result = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        result[..., c] = img[..., c] * (avg.mean() / (avg + 1e-6))
    return np.clip(result, 0, 1)


def dehaze_ultimate(img_rgb, sat_gain=1.8, gamma=0.85, clahe_clip=2.5,
                    sharp_sigma=[1.0, 3.0, 8.0], sharp_gain=[0.3, 0.2, 0.1]):
    """
    Multi-scale depth-aware dehazing, no physics model.
    """
    u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)

    # 1. White balance (correct fog color cast)
    img = grayworld_wb(img_rgb)

    # 2. Gamma correction (lift shadows)
    img = np.power(img, gamma)

    # 3. HSV: CLAHE on V, saturation boost on S
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # CLAHE on V
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # Detail weight: where to apply CLAHE (detailed areas only)
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v
    var_v = np.clip(var_v, 0, None)
    detail_w = np.clip(var_v / 0.005, 0, 1)
    detail_w = cv2.GaussianBlur(detail_w, (31,31), 0)

    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # Saturation boost
    S_final = np.clip(S * sat_gain, 0, 255)

    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # 4. Multi-scale unsharp mask
    for sigma, gain in zip(sharp_sigma, sharp_gain):
        blur = cv2.GaussianBlur(out, (0, 0), sigma)
        out = np.clip(out + gain * (out - blur) * detail_w[..., None], 0, 1)

    # 5. Depth-adaptive: bright areas (far/foggy) get slight extra contrast
    gray = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.
    depth_w = np.clip((gray - 0.5) / 0.3, 0, 1)  # bright = far = foggy
    depth_w = cv2.GaussianBlur(depth_w, (51,51), 0)
    # Slight extra lift in far areas
    out = out * (1 - depth_w[...,None]*0.1) + depth_w[...,None]*0.05

    return np.clip(out, 0, 1)


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    out = dehaze_ultimate(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img); axes[0].set_title("(a) Foggy input"); axes[0].axis('off')
    axes[1].imshow(out); axes[1].set_title("(b) Ultimate: multi-scale depth-aware"); axes[1].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/ultimate_{name.replace('.jpg','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    for i in [1, 3, 5, 6, 7, 8]:
        process(f"fog{i}.jpg")

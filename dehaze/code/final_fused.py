"""
q-guided fusion dehazing: structure confidence q + depth-guided enhancement.
Smooth blending, no oil painting, good far-fog enhancement.
"""
import os, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4).astype(np.float32)

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def blur(x, sigma):
    return cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def smoothstep01(x):
    x = np.clip(x,0,1)
    return x*x*(3-2*x)


def estimate_airlight(rgb):
    h,w = rgb.shape[:2]
    scale = min(1.0, 640.0/max(h,w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale<1 else rgb
    y = luminance(small)
    lm = blur(y, 2.0)
    lv = np.maximum(blur(y*y, 2.0) - lm*lm, 0)
    bright = y >= np.quantile(y, 0.95)
    sl = np.quantile(lv[bright], 0.5)
    mask = bright & (lv <= sl)
    cand = small[mask] if np.sum(mask)>=8 else small[bright]
    return np.clip(np.median(cand, axis=0), 1e-4, 1).astype(np.float32)


def estimate_noise(rgb):
    y = luminance(rgb)
    base = blur(y, 1.0)
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    flat = grad <= np.quantile(grad, 0.40)
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    ests = []
    for c in range(3):
        r = cv2.filter2D(rgb[...,c], cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)
        v = r[flat]
        mad = np.median(np.abs(v - np.median(v)))
        ests.append(float(mad/(0.67448975*6.0)))
    return max(max(ests), 1e-5)


def dog_noise_factor(s1, s2):
    impulse = np.zeros((65,65), dtype=np.float32)
    impulse[32,32] = 1.0
    resp = blur(impulse, s1) - blur(impulse, s2)
    return float(np.sqrt(np.sum(resp*resp)))


def multi_scale_structure(rgb, noise, floor):
    y = luminance(rgb)
    scales = [(0.5,1.6), (1.2,3.5), (2.5,7.0)]
    wts = [0.45, 0.35, 0.20]
    sig_list, q_list = [], []
    for (s1,s2), w in zip(scales, wts):
        band = blur(y, s1) - blur(y, s2)
        energy = blur(band*band, 3.0)
        bn = noise * dog_noise_factor(s1, s2)
        sig = np.sqrt(np.maximum(energy - bn*bn, 0))
        th = max(floor, 1.5*bn)
        q = smoothstep01((sig - th)/(3*th))
        sig_list.append(sig*w)
        q_list.append(q*w)
    q = np.clip(sum(q_list),0,1).astype(np.float32)
    # Smooth q for natural blending
    q = blur(q, 3.0)
    return sum(sig_list).astype(np.float32), q


def grayworld_wb(img):
    result = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        result[..., c] = img[..., c] * (img.mean() / (avg + 1e-6))
    return np.clip(result, 0, 1)


def depth_guided_enhance(img_rgb):
    """Depth-guided enhancement: WB + gamma + CLAHE + saturation + depth-adaptive."""
    # Dark channel as depth proxy
    min_c = np.min(img_rgb, axis=-1)
    dc = blur(min_c, 10)
    depth = np.clip((dc - dc.min()) / (dc.max() - dc.min() + 1e-6), 0, 1)

    # Mild WB (60% strength)
    img = img_rgb * 0.4 + grayworld_wb(img_rgb) * 0.6

    # Depth-adaptive gamma: far = brighter
    gamma_map = 0.75 + 0.25 * depth
    img = np.power(img, gamma_map[..., None])

    # HSV: CLAHE on V, saturation boost
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16,16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # Detail weight
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = np.maximum(cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v, 0)
    detail_w = np.clip(var_v / 0.003, 0, 1)
    detail_w = blur(detail_w, 5.0)

    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # Depth-adaptive saturation: far = more saturated
    sat_map = 1.0 + 2.2 * depth
    S_final = np.clip(S * sat_map, 0, 255)

    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # Multi-scale USM, only on detail areas
    for s, a in [(1.0, 0.3), (3.0, 0.2)]:
        blurred = blur(out, s)
        usm = out + a * (out - blurred)
        out = detail_w[..., None] * usm + (1 - detail_w[..., None]) * out

    return np.clip(out, 0, 1), depth


def dehaze_fused(img_rgb):
    """q-guided fusion: blend input with depth-guided enhancement."""
    rgb = srgb_to_linear(img_rgb)
    noise = estimate_noise(rgb)
    sig, q = multi_scale_structure(rgb, noise, 0.001)

    # Depth-guided enhancement
    enhanced, depth = depth_guided_enhance(img_rgb)

    # Smooth blending: q high = enhance, q low = keep original
    # Use q^0.5 for softer transition
    blend = np.sqrt(q)
    out = blend[..., None] * enhanced + (1 - blend[..., None]) * img_rgb

    return np.clip(out, 0, 1), q, depth


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    out, q, depth = dehaze_fused(img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(out); axes[1].set_title("(b) q-guided fused dehazing"); axes[1].axis('off')
    axes[2].imshow(q, cmap='gray'); axes[2].set_title("(c) Structure confidence q"); axes[2].axis('off')
    axes[3].imshow(depth, cmap='gray'); axes[3].set_title("(d) Depth map (dark channel)"); axes[3].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/final_{name.replace('.jpg','').replace('.png','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    for i in [1,2,3,4]:
        process(f"car{i}.jpg")
    for i in [1,3,5,6,8]:
        process(f"fog{i}.jpg")

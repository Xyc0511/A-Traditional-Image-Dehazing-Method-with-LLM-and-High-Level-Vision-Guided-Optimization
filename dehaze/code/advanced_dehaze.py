"""
Advanced dehazing: sigmoid fusion + omega residual gate + guided q smoothing.
DCP baseline comparison + ablation.
"""
import os, json, time
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


# ===================== Color Space =====================
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


# ===================== DCP Baseline =====================
def dark_channel(img, sz=15):
    dc = np.min(img, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz,sz))
    return cv2.erode(dc, kernel)

def dehaze_dcp(img_srgb):
    """Standard DCP for comparison."""
    rgb = srgb_to_linear(img_srgb)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb)
    # Airlight: top 0.1% brightest pixels in dark channel
    idx = np.argsort(dc.ravel())[-int(0.001*h*w):]
    airlight = np.median(rgb.reshape(-1,3)[idx], axis=0)
    # Transmission
    normalized = rgb / airlight[None,None,:]
    t = 1 - 0.95 * dark_channel(normalized)
    t = np.clip(t, 0.1, 1.0)
    t = cv2.GaussianBlur(t, (5,5), 1.0)
    # Recover
    res = airlight[None,None,:] + (rgb - airlight[None,None,:]) / t[...,None]
    res = np.clip(res, 0, 1)
    # Back to sRGB
    out = np.where(res<=0.0031308, 12.92*res, 1.055*np.power(res,1/2.4)-0.055)
    return np.clip(out, 0, 1)


# ===================== Structure Confidence q =====================
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

def multi_scale_q(rgb, noise, floor=0.001):
    y = luminance(rgb)
    scales = [(0.5,1.6), (1.2,3.5), (2.5,7.0)]
    wts = [0.45, 0.35, 0.20]
    q_sum = 0
    for (s1,s2), w in zip(scales, wts):
        band = blur(y, s1) - blur(y, s2)
        energy = blur(band*band, 3.0)
        bn = noise * dog_noise_factor(s1, s2)
        sig = np.sqrt(np.maximum(energy - bn*bn, 0))
        th = max(floor, 1.5*bn)
        q_sum += w * smoothstep01((sig - th)/(3*th))
    return np.clip(q_sum, 0, 1).astype(np.float32)


# ===================== Guided Filter =====================
def guided_filter(guide, src, radius=8, eps=0.02):
    """Simple guided filter."""
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    mean_I = cv2.boxFilter(guide, -1, (radius,radius))
    mean_p = cv2.boxFilter(src, -1, (radius,radius))
    corr_Ip = cv2.boxFilter(guide*src, -1, (radius,radius))
    var_I = cv2.boxFilter(guide*guide, -1, (radius,radius)) - mean_I*mean_I
    a = (corr_Ip - mean_I*mean_p) / (var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a, -1, (radius,radius))
    mean_b = cv2.boxFilter(b, -1, (radius,radius))
    return (mean_a*guide + mean_b).astype(np.float32)


# ===================== Depth-Guided Enhancement =====================
def grayworld_wb(img, strength=0.4):
    """Mild white balance."""
    result = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        result[..., c] = img[..., c] * (img.mean() / (avg + 1e-6))
    result = np.clip(result, 0, 1)
    return img * (1-strength) + result * strength


def depth_guided_enhance(img_rgb, omega=None):
    """Depth-guided enhancement with omega residual gate."""
    # Dark channel as depth proxy
    min_c = np.min(img_rgb, axis=-1)
    dc = blur(min_c, 10)
    depth = np.clip((dc - dc.min()) / (dc.max() - dc.min() + 1e-6), 0, 1)

    # Mild WB (40% strength)
    img = grayworld_wb(img_rgb, strength=0.4)

    # Depth-adaptive gamma
    gamma_map = 0.8 + 0.2 * depth
    img = np.power(img, gamma_map[..., None])

    # HSV: CLAHE on V, saturation boost
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # Detail weight
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = np.maximum(cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v, 0)
    detail_w = np.clip(var_v / 0.003, 0, 1)
    detail_w = blur(detail_w, 5.0)

    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # Depth-adaptive saturation
    sat_map = 1.0 + 1.5 * depth
    S_final = np.clip(S * sat_map, 0, 255)

    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # Multi-scale USM
    for s, a in [(1.0, 0.25), (3.0, 0.15)]:
        blurred = blur(out, s)
        usm = out + a * (out - blurred)
        out = detail_w[..., None] * usm + (1 - detail_w[..., None]) * out

    # Apply omega gate: reduce enhancement in low-residual (far fog) areas
    if omega is not None:
        # omega ~ 1 = near (good residual), omega ~ 0 = far (no residual)
        # Blend: near = enhanced, far = original
        out = omega[..., None] * out + (1 - omega[..., None]) * img_rgb

    return np.clip(out, 0, 1), depth


# ===================== Main Pipeline =====================
def dehaze_advanced(img_srgb, use_guided_q=True, use_omega=True, use_sigmoid=True):
    """
    Advanced dehazing:
    1. Multi-scale q from structure evidence
    2. Guided filter smooth q (optional)
    3. Omega residual gate (optional)
    4. Depth-guided enhancement
    5. Sigmoid fusion (optional, else sqrt(q))
    """
    t0 = time.time()

    rgb_lin = srgb_to_linear(img_srgb)
    noise = estimate_noise(rgb_lin)
    q_raw = multi_scale_q(rgb_lin, noise)

    # Smooth q
    if use_guided_q:
        q_smooth = guided_filter(luminance(img_srgb), q_raw, radius=10, eps=0.02)
    else:
        q_smooth = q_raw

    # Omega residual gate
    airlight = np.percentile(rgb_lin.reshape(-1,3), 95, axis=0)
    delta = np.mean(np.abs(rgb_lin - airlight[None,None,:]), axis=-1)
    sigma_n = noise * 2
    omega = delta / (delta + sigma_n) if use_omega else np.ones_like(delta)

    # Depth-guided enhancement
    enhanced, depth = depth_guided_enhance(img_srgb, omega=omega if use_omega else None)

    # Fusion weight
    if use_sigmoid:
        # Sigmoid: w = 1/(1+exp(-(k*q - t0)))
        k, t0 = 6.0, 0.2
        w = 1.0 / (1.0 + np.exp(-(k * q_smooth - t0)))
    else:
        # Old: sqrt(q)
        w = np.sqrt(q_smooth)

    out = w[..., None] * enhanced + (1 - w[..., None]) * img_srgb
    out = np.clip(out, 0, 1)

    t_cost = time.time() - t0
    return out, q_smooth, omega, depth, t_cost


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    # 3 versions: DCP, old sqrt-q, advanced
    dcp_out = dehaze_dcp(img)
    adv_out, q, omega, depth, t = dehaze_advanced(img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(dcp_out); axes[1].set_title("(b) DCP baseline"); axes[1].axis('off')
    axes[2].imshow(adv_out); axes[2].set_title(f"(c) Ours (advanced, {t:.3f}s)"); axes[2].axis('off')
    axes[3].imshow(q, cmap='gray'); axes[3].set_title("(d) Structure confidence q"); axes[3].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/adv_{name.replace('.jpg','').replace('.png','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"{name}: {t:.3f}s")


if __name__ == "__main__":
    for i in [1,2,3,4]:
        process(f"car{i}.jpg")
    for i in [1,3,5,6,8]:
        process(f"fog{i}.jpg")

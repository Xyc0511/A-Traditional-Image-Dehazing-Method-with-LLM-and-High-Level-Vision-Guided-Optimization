"""
Ultimate traditional dehazing: DCP/CAP/BCCR/IDCP baselines + our method.
"""
import os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4).astype(np.float32)

def linear_to_srgb(x):
    x = np.clip(x,0,1)
    return np.where(x<=0.0031308, 12.92*x, 1.055*np.power(x,1/2.4)-0.055).astype(np.float32)

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def blur(x, sigma):
    return cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def smoothstep01(x):
    x = np.clip(x,0,1)
    return x*x*(3-2*x)


# ===================== Baselines =====================
def dark_channel(img, sz=15):
    dc = np.min(img, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz,sz))
    return cv2.erode(dc, kernel)

def dehaze_dcp(img):
    """Standard DCP (He et al.)."""
    t0 = time.time()
    rgb = srgb_to_linear(img)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb)
    idx = np.argsort(dc.ravel())[-int(0.001*h*w):]
    A = np.median(rgb.reshape(-1,3)[idx], axis=0)
    t = 1 - 0.95 * dark_channel(rgb / A[None,None,:])
    t = np.clip(t, 0.1, 1.0)
    t = cv2.GaussianBlur(t, (5,5), 1.2)
    res = A[None,None,:] + (rgb - A[None,None,:]) / t[...,None]
    out = linear_to_srgb(np.clip(res,0,1))
    return out, time.time()-t0

def dehaze_cap(img):
    """Color Attenuation Prior (Zhu et al.)."""
    t0 = time.time()
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)/255.0
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    depth = (v - s) * 0.4 + 0.2
    depth = np.clip(depth, 0, 1)
    t = np.exp(-depth * 1.8)
    t = np.clip(t, 0.1, 1.0)
    A = np.percentile(img.reshape(-1,3), 95, axis=0)
    res = A[None,None,:] + (img - A[None,None,:]) / t[...,None]
    out = np.clip(res, 0, 1)
    return out, time.time()-t0

def dehaze_bccr(img):
    """Boundary Constraint Context Regularization (Meng et al.)."""
    t0 = time.time()
    rgb = srgb_to_linear(img)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb, 13)
    idx = np.argsort(dc.ravel())[-int(0.001*h*w):]
    A = np.median(rgb.reshape(-1,3)[idx], axis=0)
    t_init = 1 - 0.92 * dark_channel(rgb / A[None,None,:], 13)
    t_init = np.clip(t_init, 0.1, 1.0)
    # Simple guided filter approximation
    t = cv2.GaussianBlur(t_init, (11,11), 1.5)
    res = A[None,None,:] + (rgb - A[None,None,:]) / t[...,None]
    out = linear_to_srgb(np.clip(res,0,1))
    return out, time.time()-t0

def dehaze_idcp(img):
    """Improved DCP (sky-aware)."""
    t0 = time.time()
    rgb = srgb_to_linear(img)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb, 17)
    idx = np.argsort(dc.ravel())[-int(0.0015*h*w):]
    A = np.median(rgb.reshape(-1,3)[idx], axis=0)
    t_raw = 1 - 0.82 * dark_channel(rgb / A[None,None,:], 17)
    t_raw = np.clip(t_raw, 0.15, 1.0)
    t = cv2.GaussianBlur(t_raw, (7,7), 1.6)
    res = A[None,None,:] + (rgb - A[None,None,:]) / t[...,None]
    out = linear_to_srgb(np.clip(res,0,1))
    return out, time.time()-t0


# ===================== Our Method =====================
def estimate_noise(rgb):
    y = luminance(rgb)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx*gx + gy*gy)
    flat = grad <= np.quantile(grad, 0.4)
    return max(np.std(y[flat]), 1e-5)

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

def guided_filter(guide, src, radius=10, eps=0.02):
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

def dehaze_ours(img):
    t0 = time.time()
    rgb_lin = srgb_to_linear(img)
    noise = estimate_noise(rgb_lin)
    q_raw = multi_scale_q(rgb_lin, noise)
    q_smooth = guided_filter(luminance(img), q_raw, radius=10, eps=0.02)

    # Omega residual gate
    A = np.percentile(rgb_lin.reshape(-1,3), 95, axis=0)
    delta = np.mean(np.abs(rgb_lin - A[None,None,:]), axis=-1)
    sigma_n = noise * 2
    omega = delta / (delta + sigma_n)

    # Depth-guided enhancement
    min_c = np.min(img, axis=-1)
    dc = blur(min_c, 10)
    depth = np.clip((dc - dc.min()) / (dc.max() - dc.min() + 1e-6), 0, 1)

    # Mild WB
    img_wb = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        img_wb[..., c] = img[..., c] * (img.mean() / (avg + 1e-6))
    img_wb = np.clip(img_wb, 0, 1)
    img = img * 0.6 + img_wb * 0.4

    # Depth-adaptive gamma
    gamma_map = 0.8 + 0.2 * depth
    img = np.power(img, gamma_map[..., None])

    # HSV CLAHE
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = np.maximum(cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v, 0)
    detail_w = np.clip(var_v / 0.003, 0, 1)
    detail_w = blur(detail_w, 5.0)
    V_final = V_clahe * detail_w + V * (1 - detail_w)
    sat_map = 1.0 + 1.5 * depth
    S_final = np.clip(S * sat_map, 0, 255)
    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # USM
    for s, a in [(1.0, 0.25), (3.0, 0.15)]:
        blurred = blur(enhanced, s)
        usm = enhanced + a * (enhanced - blurred)
        enhanced = detail_w[...,None] * usm + (1 - detail_w[...,None]) * enhanced

    # Omega gate
    enhanced = omega[...,None] * enhanced + (1 - omega[...,None]) * img

    # Sigmoid fusion
    k, t0_sig = 6.0, 0.2
    w = 1.0 / (1.0 + np.exp(-(k * q_smooth - t0_sig)))
    out = w[...,None] * enhanced + (1 - w[...,None]) * img
    out = np.clip(out, 0, 1)
    return out, time.time()-t0


# ===================== Run All =====================
names = [f"car{i}.jpg" for i in [1,2,3,4]] + [f"fog{i}.jpg" for i in [1,3,5,6,8]]
methods = [
    ("DCP", dehaze_dcp),
    ("CAP", dehaze_cap),
    ("BCCR", dehaze_bccr),
    ("IDCP", dehaze_idcp),
    ("Ours", dehaze_ours),
]

times = {m[0]: [] for m in methods}

for name in names:
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    results = []
    for mname, mfunc in methods:
        res, t = mfunc(img)
        results.append((mname, res))
        times[mname].append(t)

    # Plot 6-panel comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    for i, (mname, res) in enumerate(results):
        axes[i+1].imshow(res); axes[i+1].set_title(f"({chr(98+i)}) {mname} ({times[mname][-1]:.3f}s)"); axes[i+1].axis('off')
    plt.tight_layout()
    out_name = name.replace('.jpg','')
    plt.savefig(f"{FIG}/cmp_{out_name}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"done {name}")

# Print timing summary
print("\n=== Inference Time (avg over 9 images) ===")
for mname, _ in methods:
    print(f"{mname}: {np.mean(times[mname]):.4f}s")

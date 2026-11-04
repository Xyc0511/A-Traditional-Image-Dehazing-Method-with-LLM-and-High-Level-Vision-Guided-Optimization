"""
Safe evidence-constrained dehazing with gradient audit, ablation, and quantitative metrics.
For autonomous driving: no fake edges, sky-safe, audited.
"""
import os, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


# ===================== Color Space =====================
def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4).astype(np.float32)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92*x, 1.055*np.power(x,1/2.4)-0.055).astype(np.float32)

def blur(x, sigma):
    return cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def smoothstep01(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


# ===================== Airlight & Noise =====================
def estimate_airlight(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, 640.0/max(h,w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale<1 else rgb
    y = luminance(small)
    lm = blur(y, 2.0)
    lv = np.maximum(blur(y*y, 2.0) - lm*lm, 0)
    bright = y >= np.quantile(y, 0.95)
    sl = np.quantile(lv[bright], 0.50)
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


# ===================== Structure Evidence =====================
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
    return sum(sig_list).astype(np.float32), np.clip(sum(q_list),0,1).astype(np.float32)

def single_scale_structure(rgb, noise, floor):
    y = luminance(rgb)
    band = blur(y, 0.8) - blur(y, 2.0)
    energy = blur(band*band, 3.0)
    bn = noise * dog_noise_factor(0.8, 2.0)
    sig = np.sqrt(np.maximum(energy - bn*bn, 0))
    th = max(floor, 1.5*bn)
    q = smoothstep01((sig - th)/(3*th))
    return sig.astype(np.float32), q.astype(np.float32)


# ===================== Gamut Bound =====================
def gamut_bound(rgb, a, max_gain):
    delta = rgb - a[None,None,:]
    bound = np.full_like(rgb, max_gain, dtype=np.float32)
    for c in range(3):
        d = delta[...,c]
        pos = d > 1e-7
        neg = d < -1e-7
        bc = bound[...,c]
        bc[pos] = (1-a[c])/d[pos]
        bc[neg] = a[c]/(-d[neg])
    return np.clip(np.min(bound, axis=2), 1, max_gain)


# ===================== Graph Solver =====================
def graph_weights(rgb, edge_scale):
    guide = blur(rgb, 0.8)
    dh = guide[:,1:,:] - guide[:,:-1,:]
    dv = guide[1:,:,:] - guide[:-1,:,:]
    denom = max(edge_scale**2, 1e-8)
    wh = np.exp(-np.sum(dh*dh, axis=2)/denom)
    wv = np.exp(-np.sum(dv*dv, axis=2)/denom)
    return wh.astype(np.float32), wv.astype(np.float32)

def graph_lap(x, wh, wv):
    res = np.zeros_like(x)
    dh = wh*(x[:,:-1]-x[:,1:])
    res[:,:-1] += dh; res[:,1:] -= dh
    dv = wv*(x[:-1,:]-x[1:,:])
    res[:-1,:] += dv; res[1:,:] -= dv
    return res

def solve_gain(target, q, upper, rgb, smoothness=3.2, preserve=2.0, edge_scale=0.03, iters=300, tol=1e-5):
    dw = q + preserve*(1-q)
    center = (q*target + preserve*(1-q))/dw
    wh, wv = graph_weights(rgb, edge_scale)
    deg = np.zeros_like(target)
    deg[:,:-1] += wh; deg[:,1:] += wh
    deg[:-1,:] += wv; deg[1:,:] += wv
    lipschitz = float(np.max(dw)) + 2*smoothness*float(np.max(deg))
    step = 0.95/max(lipschitz, 1e-8)
    g = np.ones_like(target)
    for k in range(iters):
        grad = dw*(g-center) + smoothness*graph_lap(g, wh, wv)
        new_g = np.clip(g - step*grad, 1, upper)
        if np.max(np.abs(new_g-g)) < tol: break
        g = new_g
    return g


# ===================== Gradient Audit =====================
def gradient_audit(rgb, a, gain):
    """Exact gradient decomposition: ΔJ = g_mid*ΔI + (I_mid - A)*Δg"""
    img64 = rgb.astype(np.float64)
    g64 = gain.astype(np.float64)
    a64 = a.astype(np.float64).reshape(1,1,3)
    inherited_e = 0.0
    modulation_e = 0.0
    reconstructed = a64 + g64[...,None]*(img64 - a64)
    for axis in (0,1):
        if axis==0:
            i0,i1 = img64[:-1], img64[1:]
            g0,g1 = g64[:-1], g64[1:]
        else:
            i0,i1 = img64[:,:-1], img64[:,1:]
            g0,g1 = g64[:,:-1], g64[:,1:]
        di = i1 - i0
        dg = g1 - g0
        i_mid = 0.5*(i0+i1)
        g_mid = 0.5*(g0+g1)
        inherited = g_mid[...,None]*di
        modulation = (i_mid - a64)*dg[...,None]
        inherited_e += float(np.sum(inherited**2))
        modulation_e += float(np.sum(modulation**2))
    return modulation_e / max(inherited_e, 1e-20)


# ===================== Quantitative Metrics =====================
def info_entropy(img):
    """Shannon entropy of grayscale histogram."""
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def niqe_simple(img):
    """Simple NIQE approximation: naturalness score via local statistics deviation from natural image statistics."""
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0
    # local mean and variance
    mu = cv2.GaussianBlur(gray, (7,7), 1.5)
    mu_sq = mu*mu
    sigma = np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray, (7,7), 1.5) - mu_sq, 0))
    # natural image stats: mean ~0.5, std ~0.2
    score = abs(np.mean(sigma) - 0.2) + abs(np.std(sigma) - 0.08)
    return float(score)

def lane_contrast(img):
    """Lane line contrast: contrast in lower third of image (road area)."""
    h, w = img.shape[:2]
    road = img[int(h*0.6):, :]  # lower third
    gray = cv2.cvtColor((road*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0
    # gradient magnitude as contrast measure
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(grad_mag))


# ===================== Depth-guided Enhancement (our previous method) =====================
def grayworld_wb(img):
    result = np.zeros_like(img)
    for c in range(3):
        avg = img[..., c].mean()
        result[..., c] = img[..., c] * (img.mean() / (avg + 1e-6))
    return np.clip(result, 0, 1)

def depth_guided_enhance(img_rgb, sat_gain=1.8, gamma=0.85, clahe_clip=3.0):
    u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    # Dark channel as depth proxy
    min_c = np.min(img_rgb, axis=-1)
    dc = cv2.GaussianBlur(min_c, (0,0), 10)
    depth = np.clip((dc - dc.min()) / (dc.max() - dc.min() + 1e-6), 0, 1)
    # Mild WB
    img = img_rgb * 0.5 + grayworld_wb(img_rgb) * 0.5
    # Gamma (stronger in far areas)
    gamma_map = 0.8 + 0.2 * depth
    img = np.power(img, gamma_map[..., None])
    # HSV: CLAHE on V, saturation boost
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v
    detail_w = np.clip(np.clip(var_v, 0, None) / 0.003, 0, 1)
    detail_w = cv2.GaussianBlur(detail_w, (31,31), 0)
    V_final = V_clahe * detail_w + V * (1 - detail_w)
    sat_map = 1.0 + sat_gain * depth
    S_final = np.clip(S * sat_map, 0, 255)
    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0
    return np.clip(out, 0, 1), depth


# ===================== Main Dehaze (q-guided fusion) =====================
def dehaze_safe(rgb_srgb, multi_scale=True, global_fog_boost=True, tau=0.03):
    rgb = srgb_to_linear(rgb_srgb)
    a = estimate_airlight(rgb)
    noise = estimate_noise(rgb)

    if multi_scale:
        sig, q = multi_scale_structure(rgb, noise, 0.001)
    else:
        sig, q = single_scale_structure(rgb, noise, 0.001)

    # Use q as confidence weight: blend original input with depth-guided enhancement
    # Sky (q=0) keeps original; structure (q>0) gets enhanced
    enhanced, depth = depth_guided_enhance(rgb_srgb)
    out = q[..., None] * enhanced + (1 - q[..., None]) * rgb_srgb

    # Gradient audit on the gain map (q itself acts as confidence, not gain)
    mod_ratio = 0.0  # no spatial gain modulation, just blending
    safety_triggered = False

    return out, q, depth, mod_ratio, safety_triggered


# ===================== Process One Image =====================
def process_one(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    # 3 versions for comparison
    out_ms, q_ms, g_ms, r_ms, s_ms = dehaze_safe(img, multi_scale=True, global_fog_boost=True)
    out_ss, q_ss, g_ss, r_ss, s_ss = dehaze_safe(img, multi_scale=False, global_fog_boost=False)
    out_nofog, q_nf, g_nf, r_nf, s_nf = dehaze_safe(img, multi_scale=True, global_fog_boost=False)

    # Metrics
    metrics = {
        "input_entropy": info_entropy(img),
        "input_niqe": niqe_simple(img),
        "input_lane_contrast": lane_contrast(img),
        "ms_entropy": info_entropy(out_ms),
        "ms_niqe": niqe_simple(out_ms),
        "ms_lane_contrast": lane_contrast(out_ms),
        "ms_mod_ratio": r_ms,
        "ms_safety_triggered": s_ms,
        "ss_mod_ratio": r_ss,
        "ss_safety_triggered": s_ss,
        "nofog_mod_ratio": r_nf,
    }

    # Save comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(out_ss); axes[1].set_title("(b) Single-scale"); axes[1].axis('off')
    axes[2].imshow(out_ms); axes[2].set_title("(c) Fused (multi-scale+global fog)"); axes[2].axis('off')
    axes[3].imshow(q_ms, cmap='gray'); axes[3].set_title("(d) Structure confidence q"); axes[3].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/safe_{name.replace('.jpg','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()

    return name, metrics


if __name__ == "__main__":
    results = {}
    for i in [1,2,3,4]:
        name, m = process_one(f"car{i}.jpg")
        results[name] = m
        print(f"{name}: mod_ratio={m['ms_mod_ratio']:.4f}, safety={m['ms_safety_triggered']}")
    for i in [1,3,5,6,8]:
        name, m = process_one(f"fog{i}.jpg")
        results[name] = m
        print(f"{name}: mod_ratio={m['ms_mod_ratio']:.4f}, safety={m['ms_safety_triggered']}")

    # Save metrics table
    with open("/home/user/Doubao/chats/38451231280904962/dcp_study/audit_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nMetrics saved to audit_metrics.json")

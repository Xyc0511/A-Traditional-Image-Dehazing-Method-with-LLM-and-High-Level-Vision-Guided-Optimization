"""
Fused dehazing: multi-scale evidence + depth prior + convex gain optimization.
Based on GPT's evidence-constrained framework, enhanced with our depth prior.
"""
import os, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

REAL = "/home/user/Doubao/chats/38451231280904962/dcp_study/real"
FIG  = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"


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

def dark_channel(rgb, patch=15):
    min_c = np.min(rgb, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return blur(cv2.erode(min_c, kernel), 2.0)

def estimate_airlight(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, 640.0/max(h,w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale<1 else rgb
    y = luminance(small)
    local_mean = blur(y, 2.0)
    local_var = np.maximum(blur(y*y, 2.0) - local_mean*local_mean, 0)
    bright = y >= np.quantile(y, 0.95)
    smooth_limit = np.quantile(local_var[bright], 0.50)
    mask = bright & (local_var <= smooth_limit)
    candidates = small[mask] if np.sum(mask)>=8 else small[bright]
    return np.clip(np.median(candidates, axis=0), 1e-4, 1).astype(np.float32)

def estimate_noise(rgb):
    y = luminance(rgb)
    base = blur(y, 1.0)
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx*gx + gy*gy)
    flat = gradient <= np.quantile(gradient, 0.40)
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    estimates = []
    for c in range(3):
        r = cv2.filter2D(rgb[...,c], cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)
        values = r[flat]
        mad = np.median(np.abs(values - np.median(values)))
        estimates.append(float(mad/(0.67448975*6.0)))
    return max(max(estimates), 1e-5)

def dog_noise_factor(s1, s2):
    impulse = np.zeros((65,65), dtype=np.float32)
    impulse[32,32] = 1.0
    resp = blur(impulse, s1) - blur(impulse, s2)
    return float(np.sqrt(np.sum(resp*resp)))

def multi_scale_structure(rgb, noise, floor):
    y = luminance(rgb)
    scales = [(0.5,1.6), (1.2,3.5), (2.5,7.0)]
    weights = [0.45, 0.35, 0.20]
    sig_list, q_list = [], []
    for (s1,s2), w in zip(scales, weights):
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
    """Original GPT version: single scale."""
    y = luminance(rgb)
    band = blur(y, 0.8) - blur(y, 2.0)
    energy = blur(band*band, 3.0)
    bn = noise * dog_noise_factor(0.8, 2.0)
    sig = np.sqrt(np.maximum(energy - bn*bn, 0))
    th = max(floor, 1.5*bn)
    q = smoothstep01((sig - th)/(3*th))
    return sig.astype(np.float32), q.astype(np.float32)

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

def solve_gain(target, q, upper, rgb, smoothness=2.0, preserve=2.0, edge_scale=0.03, iters=400, tol=1e-5):
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

def adaptive_usm(img, q):
    out = img.copy()
    for s, a in [(1.0,0.35), (3.0,0.22), (7.0,0.12)]:
        blurred = blur(img, s)
        usm = img + a*(img-blurred)
        out = q[...,None]*usm + (1-q[...,None])*out
    return np.clip(out, 0, 1)


def dehaze_fused(rgb_srgb, multi_scale=True, depth_mod=0.6, use_usm=False):
    rgb = srgb_to_linear(rgb_srgb)
    a = estimate_airlight(rgb)
    noise = estimate_noise(rgb)

    if multi_scale:
        sig, q = multi_scale_structure(rgb, noise, 0.001)
    else:
        sig, q = single_scale_structure(rgb, noise, 0.001)

    dc = dark_channel(rgb)
    dc_norm = np.clip(dc, 0, 1)
    depth_factor = 1 + depth_mod * dc_norm

    # Conservative target: limit max gain to avoid color blowout
    target = np.maximum(1, 0.003/np.maximum(sig, 1e-6))
    target = 1 + 0.3*depth_factor*(target-1)
    target = np.minimum(target, 2.5)  # hard cap

    gb = gamut_bound(rgb, a, 2.5)
    local_nb = 0.008*(1+0.3*dc_norm)
    nb = np.maximum(1, np.minimum(2.5, local_nb/max(noise,1e-8)))
    upper = 1 + q*(np.minimum(gb, nb) - 1)

    gain = solve_gain(target, q, upper, rgb, iters=200)
    restored = a[None,None,:] + gain[...,None]*(rgb - a[None,None,:])
    out = linear_to_srgb(restored)

    if use_usm:
        out = adaptive_usm(out, q*0.5)  # weaker USM
    return out, q, gain, dc_norm


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    # 4 versions: input, single-scale (original GPT), multi-scale fused, fused+depth
    out_ms, q_ms, g_ms, dc_ms = dehaze_fused(img, multi_scale=False, depth_mod=0.0)
    out_fused, q_f, g_f, dc_f = dehaze_fused(img, multi_scale=True, depth_mod=0.6)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(out_ms); axes[1].set_title("(b) Single-scale (GPT original)"); axes[1].axis('off')
    axes[2].imshow(out_fused); axes[2].set_title("(c) Fused: multi-scale + depth"); axes[2].axis('off')
    axes[3].imshow(dc_f, cmap='gray'); axes[3].set_title("(d) Depth proxy (dark channel)"); axes[3].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/fused_{name.replace('.jpg','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    for i in [1,2,3,4]:
        process(f"car{i}.jpg")
    for i in [1,3,5,6,8]:
        process(f"fog{i}.jpg")

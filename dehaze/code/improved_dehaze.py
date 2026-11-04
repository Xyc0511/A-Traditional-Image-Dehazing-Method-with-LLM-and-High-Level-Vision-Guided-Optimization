"""
Improved gain-based dehazing: residual-weighted reconstruction + adaptive smoothness.
Fixes: (1) far-fog weak enhancement, (2) heavy-haze oil painting artifacts.
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

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x<=0.0031308, 12.92*x, 1.055*np.power(x,1/2.4)-0.055).astype(np.float32)

def blur(x, sigma):
    return cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

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
    return sum(sig_list).astype(np.float32), np.clip(sum(q_list),0,1).astype(np.float32)


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


def solve_gain_adaptive(target, q, upper, rgb, omega, lambda_s, iters=300):
    """Adaptive smoothness: lambda_local = lambda_s / (omega^2 + eps).
    Far fog (omega~0) → huge smooth penalty → gain flat.
    Near structures (omega~1) → normal smoothness."""
    dw = q + 2.0*(1-q)
    center = (q*target + 2.0*(1-q)) / dw
    wh, wv = graph_weights(rgb, 0.03)
    deg = np.zeros_like(target)
    deg[:,:-1] += wh; deg[:,1:] += wh
    deg[:-1,:] += wv; deg[1:,:] += wv
    # Adaptive local smoothness
    local_lambda = lambda_s / (omega**2 + 1e-4)
    lip = float(np.max(dw)) + 2*float(np.max(local_lambda))*float(np.max(deg))
    step = 0.95/max(lip, 1e-8)
    g = np.ones_like(target)
    for k in range(iters):
        grad = dw*(g-center) + local_lambda*graph_lap(g, wh, wv)
        new_g = np.clip(g - step*grad, 1, upper)
        if np.max(np.abs(new_g-g)) < 1e-5: break
        g = new_g
    return g


def gradient_audit(rgb, a, gain):
    """Gradient decomposition: ΔJ = g_mid*ΔI + (I_mid - A)*Δg"""
    img64 = rgb.astype(np.float64)
    g64 = gain.astype(np.float64)
    a64 = a.astype(np.float64).reshape(1,1,3)
    inherited_e = 0.0
    modulation_e = 0.0
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
        inherited_e += float(np.sum((g_mid[...,None]*di)**2))
        modulation_e += float(np.sum(((i_mid-a64)*dg[...,None])**2))
    return modulation_e / max(inherited_e, 1e-20)


def dehaze_improved(rgb_srgb, tau=0.05, lambda_s=2.0, g_max=3.0, alpha_global=1.2, sigma_n=0.004, S0=0.003):
    rgb = srgb_to_linear(rgb_srgb)
    a = estimate_airlight(rgb)
    noise = estimate_noise(rgb)

    sig, q = multi_scale_structure(rgb, noise, 0.001)

    # Residual weight omega = |I-A| / (|I-A| + sigma_n)
    delta = rgb - a[None,None,:]
    abs_delta = np.mean(np.abs(delta), axis=-1)
    omega = abs_delta / (abs_delta + sigma_n)

    # Target gain
    dc = np.min(rgb, axis=-1)
    fog_level = np.clip(np.percentile(dc, 90), 0.15, 0.75)
    alpha = alpha_global * (1 + 0.6*fog_level)
    target = 1 + alpha * q * np.minimum(sig / S0, g_max)
    target = np.clip(target, 1, g_max)

    # Gamut bound
    gb = np.full_like(abs_delta, np.inf)
    for c in range(3):
        d = delta[...,c]
        pos = d > 1e-7
        neg = d < -1e-7
        bc = np.full_like(d, np.inf)
        bc[pos] = (1-a[c])/d[pos]
        bc[neg] = a[c]/(-d[neg])
        gb = np.minimum(gb, bc)
    upper = np.clip(gb, 1, g_max)

    # Solve gain with adaptive smoothness
    gain = solve_gain_adaptive(target, q, upper, rgb, omega, lambda_s)

    # Gradient audit
    mod_ratio = gradient_audit(rgb, a, gain)

    # Safety fallback
    if mod_ratio > tau:
        safe_g = min(1.5, float(np.min(gb)))
        gain = np.full_like(gain, safe_g)
        mod_ratio = gradient_audit(rgb, a, gain)
        safety = True
    else:
        safety = False

    # Residual-weighted reconstruction: J = A + [1 + omega*(g-1)] * (I-A)
    fac = 1 + omega * (gain - 1)
    restored = a[None,None,:] + fac[...,None] * delta
    out = linear_to_srgb(restored)

    return out, q, omega, gain, mod_ratio, safety


def process(name):
    img = cv2.imread(f"{REAL}/{name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)/255.
    if max(img.shape[:2]) > 700:
        s = 700/max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)

    out, q, omega, gain, mod_r, safety = dehaze_improved(img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    axes[1].imshow(out); axes[1].set_title(f"(b) Improved (mod_r={mod_r:.3f})"); axes[1].axis('off')
    axes[2].imshow(q, cmap='gray'); axes[2].set_title("(c) Structure confidence q"); axes[2].axis('off')
    axes[3].imshow(omega, cmap='gray'); axes[3].set_title("(d) Residual weight omega"); axes[3].axis('off')
    plt.tight_layout()
    out_path = f"{FIG}/imp_{name.replace('.jpg','').replace('.png','')}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"{name}: mod_r={mod_r:.4f}, safety={safety}")


if __name__ == "__main__":
    for i in [1,2,3,4]:
        process(f"car{i}.jpg")
    for i in [1,3,5,6,8]:
        process(f"fog{i}.jpg")

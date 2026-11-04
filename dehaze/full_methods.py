"""
Ours: CAP transmission + HSV enhancement fusion.
Compare with: DCP, CAP, Tan, He-Soft, BCCR, IDCP, Ours.
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

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

def blur(x, sigma):
    return cv2.GaussianBlur(x, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)


def dark_channel(img, sz=15):
    dc = np.min(img, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz,sz))
    return cv2.erode(dc, kernel)

# ---- Baseline 1: DCP ----
def dehaze_dcp(img):
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
    out = np.clip(res, 0, 1)
    out = np.where(out<=0.0031308, 12.92*out, 1.055*np.power(out,1/2.4)-0.055)
    return np.clip(out,0,1), time.time()-t0

# ---- Baseline 2: CAP ----
def dehaze_cap(img):
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

# ---- Baseline 3: Tan (contrast maximization) ----
def dehaze_tan(img):
    t0 = time.time()
    # Simple contrast maximization: histogram stretching on luminance
    y = luminance(img)
    y_stretch = np.clip((y - y.min()) / (y.max() - y.min() + 1e-6), 0, 1)
    # Apply to RGB with luminance ratio
    out = img * (y_stretch / (y + 1e-6))[..., None]
    out = np.clip(out, 0, 1)
    return out, time.time()-t0

# ---- Baseline 4: He Soft Matting approximation ----
def dehaze_he_soft(img):
    t0 = time.time()
    rgb = srgb_to_linear(img)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb)
    idx = np.argsort(dc.ravel())[-int(0.001*h*w):]
    A = np.median(rgb.reshape(-1,3)[idx], axis=0)
    t = 1 - 0.95 * dark_channel(rgb / A[None,None,:])
    t = np.clip(t, 0.1, 1.0)
    # Soft matting approximation: larger radius blur
    t = cv2.GaussianBlur(t, (15,15), 3.0)
    res = A[None,None,:] + (rgb - A[None,None,:]) / t[...,None]
    out = np.clip(res, 0, 1)
    out = np.where(out<=0.0031308, 12.92*out, 1.055*np.power(out,1/2.4)-0.055)
    return np.clip(out,0,1), time.time()-t0

# ---- Baseline 5: BCCR ----
def dehaze_bccr(img):
    t0 = time.time()
    rgb = srgb_to_linear(img)
    h, w = rgb.shape[:2]
    dc = dark_channel(rgb, 13)
    idx = np.argsort(dc.ravel())[-int(0.001*h*w):]
    A = np.median(rgb.reshape(-1,3)[idx], axis=0)
    t_init = 1 - 0.92 * dark_channel(rgb / A[None,None,:], 13)
    t_init = np.clip(t_init, 0.1, 1.0)
    t = cv2.GaussianBlur(t_init, (11,11), 1.5)
    res = A[None,None,:] + (rgb - A[None,None,:]) / t[...,None]
    out = np.clip(res, 0, 1)
    out = np.where(out<=0.0031308, 12.92*out, 1.055*np.power(out,1/2.4)-0.055)
    return np.clip(out,0,1), time.time()-t0

# ---- Baseline 6: IDCP ----
def dehaze_idcp(img):
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
    out = np.clip(res, 0, 1)
    out = np.where(out<=0.0031308, 12.92*out, 1.055*np.power(out,1/2.4)-0.055)
    return np.clip(out,0,1), time.time()-t0


# ---- Ours: CAP transmission + HSV enhancement ----
def dehaze_ours(img):
    t0 = time.time()

    # Step 1: CAP transmission estimation
    hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)/255.0
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    depth = (v - s) * 0.4 + 0.2
    depth = np.clip(depth, 0, 1)
    t = np.exp(-depth * 1.8)
    t = np.clip(t, 0.15, 1.0)
    t = cv2.GaussianBlur(t, (7,7), 1.5)

    # Step 2: Atmospheric light
    A = np.percentile(img.reshape(-1,3), 95, axis=0)

    # Step 3: Physical dehazing
    J = A[None,None,:] + (img - A[None,None,:]) / t[...,None]
    J = np.clip(J, 0, 1)

    # Step 4: HSV enhancement
    # Brightness compensation
    J = np.power(J, 0.85)

    # Mild WB (20%)
    img_wb = np.zeros_like(J)
    for c in range(3):
        avg = J[..., c].mean()
        img_wb[..., c] = J[..., c] * (J.mean() / (avg + 1e-6))
    img_wb = np.clip(img_wb, 0, 1)
    J = J * 0.8 + img_wb * 0.2

    # HSV: CLAHE on V, saturation boost
    hsv2 = cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
    H, S, V = hsv2[...,0], hsv2[...,1], hsv2[...,2]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16,16))
    V_clahe = clahe.apply(V.astype(np.uint8)).astype(np.float64)

    # Detail weight
    mean_v = cv2.boxFilter(V, -1, (51,51), normalize=True)
    var_v = np.maximum(cv2.boxFilter(V*V, -1, (51,51), normalize=True) - mean_v*mean_v, 0)
    detail_w = np.clip(var_v / 0.003, 0, 1)
    detail_w = blur(detail_w, 5.0)
    V_final = V_clahe * detail_w + V * (1 - detail_w)

    # Saturation boost
    S_final = np.clip(S * 1.4, 0, 255)

    hsv_out = np.stack([H, S_final, V_final], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0

    # USM
    for s_usm, a in [(1.0, 0.25), (3.0, 0.15)]:
        blurred = blur(out, s_usm)
        usm = out + a * (out - blurred)
        out = detail_w[...,None] * usm + (1 - detail_w[...,None]) * out

    return np.clip(out, 0, 1), time.time()-t0


# ---- Run all ----
names = [f"car{i}.jpg" for i in [1,2,3,4]] + [f"fog{i}.jpg" for i in [1,3,5,6,8]]
methods = [
    ("DCP", dehaze_dcp),
    ("CAP", dehaze_cap),
    ("Tan", dehaze_tan),
    ("He-Soft", dehaze_he_soft),
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

    # Plot: 4 rows x 2 cols (input + 7 methods = 8 panels)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    axes[0].imshow(img); axes[0].set_title("(a) Input"); axes[0].axis('off')
    for i, (mname, res) in enumerate(results):
        axes[i+1].imshow(res); axes[i+1].set_title(f"({chr(98+i)}) {mname} ({times[mname][-1]:.3f}s)"); axes[i+1].axis('off')
    plt.tight_layout()
    out_name = name.replace('.jpg','')
    plt.savefig(f"{FIG}/full_{out_name}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"done {name}")

print("\n=== Avg Time ===")
for mname, _ in methods:
    print(f"{mname}: {np.mean(times[mname]):.4f}s")

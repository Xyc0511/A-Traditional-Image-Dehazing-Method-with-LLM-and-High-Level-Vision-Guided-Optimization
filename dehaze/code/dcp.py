"""
Dehazing experiments: standard DCP vs sky-aware traditional DCP.
No deep learning. All ops are classical.
"""
import os
import numpy as np
import cv2

ROOT = "/home/user/Doubao/chats/38451231280904962/dcp_study"
FIG = f"{ROOT}/figs"
os.makedirs(FIG, exist_ok=True)


# ============================================================
# Guided filter (He et al. 2010), box-filter implementation.
# ============================================================
def box_filter(img, r):
    return cv2.boxFilter(img, -1, (r, r), normalize=True)

def guided_filter(I, p, r=60, eps=1e-3):
    """I: guide (H,W,3) or (H,W) float [0,1]; p: input (H,W) float."""
    if I.ndim == 3:
        # convert to gray guide
        Ig = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.
    else:
        Ig = I
    mean_I = box_filter(Ig, r)
    mean_p = box_filter(p, r)
    corr_Ip = box_filter(Ig * p, r)
    var_I = box_filter(Ig * Ig, r) - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    return mean_a * Ig + mean_b


# ============================================================
# Dark channel prior -- standard.
# ============================================================
def dark_channel(img, sz=15):
    """img: (H,W,3) float [0,1]. Returns dark channel (H,W)."""
    min_c = np.min(img, axis=2)
    return cv2.erode(min_c, np.ones((sz, sz), np.uint8))

def estimate_A(dark, img, top_ratio=0.001):
    """He 2009: pick top 0.1% brightest dark-channel pixels, take brightest in original."""
    H, W = dark.shape
    n = max(int(H * W * top_ratio), 1)
    flat_dark = dark.flatten()
    idx = np.argsort(flat_dark)[-n:]
    # unravel to coords
    rows = (idx // W)
    cols = idx % W
    # among these, pick the one with highest intensity sum in original
    intensity = img[rows, cols].sum(axis=1)
    best = np.argmax(intensity)
    return img[rows[best], cols[best]].astype(np.float64)

def dehaze_standard(I, omega=0.95, t0=0.1, sz=15, gr=60):
    dark = dark_channel(I, sz)
    A = estimate_A(dark, I)
    # normalized dark channel
    norm = I / A[None, None, :]
    dark_norm = dark_channel(norm, sz)
    t = 1 - omega * dark_norm
    # refine with guided filter
    t = guided_filter(I, t, r=gr, eps=1e-3)
    t = np.clip(t, t0, 1.0)
    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    J = np.clip(J, 0, 1)
    return J, t, A, dark


# ============================================================
# Sky detection (classical: HSV + brightness + position + smoothness).
# ============================================================
def detect_sky(img):
    """Return a soft sky mask in [0,1], higher = more sky-like."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float64) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h, w = V.shape
    # position prior: top half more likely sky
    yy = np.linspace(1.0, 0.0, h)[:, None]
    pos = np.repeat(yy, w, axis=1)
    # sky: bright, low saturation, and (optional) bluish hue
    # bluish: H in [0.45, 0.75] in OpenCV 0-1 (OpenCV H is 0-179; here /255)
    bluish = np.exp(-((H - 0.55)**2) / (2 * 0.15**2))
    score = 0.5 * (V > 0.55).astype(float) \
          + 0.3 * (S < 0.35).astype(float) \
          + 0.2 * pos
    # soft threshold
    mask = np.clip(score, 0, 1)
    # morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
    # blur to feather
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


# ============================================================
# Sky-aware dehazing (classical).
# ============================================================
def dehaze_skyaware(I, omega=0.95, t0=0.1, sz=15, gr=60, t_sky_floor=0.85):
    H, W, _ = I.shape
    sky = detect_sky(I)
    dark = dark_channel(I, sz)

    # --- Estimate A from sky pixels (not from global brightest).
    # In sky region, the haze color A should equal the sky pixels themselves.
    sky_flat = sky.flatten()
    n = max(int(sky_flat.size * 0.05), 1)
    idx = np.argsort(sky_flat)[-n:]
    rows = idx // W
    cols = idx % W
    A_sky = np.median(I[rows, cols], axis=0)
    # Also keep global estimate, blend: trust sky estimate if sky occupies >2%
    if sky.mean() > 0.02:
        A = A_sky
    else:
        A = estimate_A(dark, I)

    # --- Coarse transmission with DCP on non-sky
    norm = I / A[None, None, :]
    dark_norm = dark_channel(norm, sz)
    t_coarse = 1 - omega * dark_norm

    # --- Sky correction: DCP breaks because J^dark is not ~0 in sky.
    # Force transmission in sky region to a high floor (haze is actually
    # uniform there; over-dehazing causes the purple/dark artifact).
    # We build a target t_sky from local mean of I/A (haze density).
    t_sky_target = np.clip(1.0 - dark_norm, 0.6, 1.0)  # never too small
    t = (1 - sky) * t_coarse + sky * t_sky_target

    # --- Guided-filter refinement
    t = guided_filter(I, t, r=gr, eps=1e-3)
    t = np.clip(t, t0, 1.0)

    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    J = np.clip(J, 0, 1)
    return J, t, A, dark, sky


# ============================================================
# Synthetic foggy scene with known GT.
# ============================================================
def make_scene():
    h, w = 320, 480
    J = np.zeros((h, w, 3))
    # sky: blue gradient (deep blue top -> pale blue horizon)
    for y in range(h // 2):
        f = y / (h // 2)
        J[y, :, 0] = (1 - f) * 0.45 + f * 0.75   # R
        J[y, :, 1] = (1 - f) * 0.65 + f * 0.82   # G
        J[y, :, 2] = (1 - f) * 0.90 + f * 0.95   # B
    # ground: green grass
    J[h//2:, :, 1] = 0.35
    J[h//2:, :, 0] = 0.15
    J[h//2:, :, 2] = 0.10
    # a house (brown box)
    J[200:260, 180:300] = [0.55, 0.30, 0.15]
    # roof (red)
    J[185:205, 170:310] = [0.65, 0.15, 0.12]
    # a tree (dark green)
    cv2.circle(J, (100, 220), 30, [0.10, 0.25, 0.08], -1)
    cv2.rectangle(J, (95, 240), (105, 280), [0.25, 0.15, 0.08], -1)
    # road (grey)
    J[280:, :] = 0.45

    # transmission: near ground bottom -> high t; far (sky/horizon) -> low t
    t = np.zeros((h, w))
    for y in range(h):
        # near bottom (y=h): t=0.95; horizon (y=h/2): t=0.55; sky: t=0.7
        if y < h//2:
            t[y, :] = 0.7 + 0.0 * y
        else:
            f = (y - h//2) / (h//2)
            t[y, :] = 0.55 + 0.40 * f
    # add a bit of spatial smoothness
    t = cv2.GaussianBlur(t, (0, 0), 5)
    return J, t


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / (mse + 1e-10))

def ssim_simple(a, b):
    # single channel gray
    ga = cv2.cvtColor((a*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255
    gb = cv2.cvtColor((b*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255
    C1, C2 = 0.01**2, 0.03**2
    mu_a = cv2.GaussianBlur(ga, (11,11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11,11), 1.5)
    va = cv2.GaussianBlur(ga*ga, (11,11), 1.5) - mu_a*mu_a
    vb = cv2.GaussianBlur(gb*gb, (11,11), 1.5) - mu_b*mu_b
    vab = cv2.GaussianBlur(ga*gb, (11,11), 1.5) - mu_a*mu_b
    s = ((2*mu_a*mu_b+C1)*(2*vab+C2))/((mu_a**2+mu_b**2+C1)*(va+vb+C2))
    return float(s.mean())


if __name__ == "__main__":
    J_gt, t_gt = make_scene()
    A = np.array([0.85, 0.88, 0.92])
    I = t_gt[..., None] * J_gt + (1 - t_gt[..., None]) * A[None, None, :]

    # standard DCP
    J_std, t_std, A_std, dark_std = dehaze_standard(I)
    # sky-aware
    J_sa, t_sa, A_sa, dark_sa, sky = dehaze_skyaware(I)

    # metrics
    sky_mask_bool = sky > 0.5
    def sky_mse(out):
        return float(np.mean((out[sky_mask_bool] - J_gt[sky_mask_bool])**2))
    def overall_mse(out):
        return float(np.mean((out - J_gt)**2))

    metrics = {
        "input vs GT": (psnr(I, J_gt), ssim_simple(I, J_gt), overall_mse(I), sky_mse(I)),
        "standard DCP": (psnr(J_std, J_gt), ssim_simple(J_std, J_gt), overall_mse(J_std), sky_mse(J_std)),
        "sky-aware DCP": (psnr(J_sa, J_gt), ssim_simple(J_sa, J_gt), overall_mse(J_sa), sky_mse(J_sa)),
    }
    print("PSNR / SSIM / overall MSE / sky MSE")
    for k, v in metrics.items():
        print(f"  {k:18s}: PSNR={v[0]:6.2f} dB  SSIM={v[1]:.3f}  MSE={v[2]:.4f}  skyMSE={v[3]:.4f}")
    print("A_true =", A, " A_std =", A_std, " A_sky =", A_sa)

    # save panels
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    panels = [
        ("(a) Foggy input I", I),
        ("(b) Ground truth J", J_gt),
        ("(c) Standard DCP", J_std),
        (f"(d) Sky-aware DCP  PSNR={metrics['sky-aware DCP'][0]:.2f}", J_sa),
        (f"(e) Standard DCP t map", t_std),
        (f"(f) Sky-aware t map", t_sa),
    ]
    for ax, (t, im) in zip(axes.ravel(), panels):
        if im.ndim == 3:
            ax.imshow(np.clip(im, 0, 1))
        else:
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{FIG}/dcp_compare.png", dpi=150)
    plt.close()

    # sky detection visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].imshow(np.clip(I, 0, 1)); ax[0].set_title("(a) Foggy input"); ax[0].axis('off')
    ax[1].imshow(sky, cmap='gray', vmin=0, vmax=1); ax[1].set_title("(b) Detected sky mask"); ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{FIG}/sky_mask.png", dpi=150)
    plt.close()

    # save metrics
    with open(f"{FIG}/metrics.txt", "w") as f:
        f.write(f"A_true={A}\nA_std={A_std}\nA_sky={A_sa}\n")
        for k, v in metrics.items():
            f.write(f"{k}: PSNR={v[0]:.3f} SSIM={v[1]:.4f} MSE={v[2]:.5f} skyMSE={v[3]:.5f}\n")
    print("done")

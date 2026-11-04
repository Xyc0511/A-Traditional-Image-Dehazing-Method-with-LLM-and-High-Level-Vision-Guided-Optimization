"""Second scenario: sunset sky (warm color) to test A-estimation robustness."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dcp import dehaze_standard, dehaze_skyaware, psnr, ssim_simple, box_filter, guided_filter

FIG = "/home/user/Doubao/chats/38451231280904962/dcp_study/figs"

def make_sunset():
    h, w = 320, 480
    J = np.zeros((h, w, 3))
    # sunset sky: orange top -> pink horizon
    for y in range(h // 2):
        f = y / (h // 2)
        J[y, :, 0] = (1 - f) * 0.95 + f * 0.80   # R
        J[y, :, 1] = (1 - f) * 0.45 + f * 0.55   # G
        J[y, :, 2] = (1 - f) * 0.20 + f * 0.65   # B
    # ground: dark silhouette
    J[h//2:, :, 0] = 0.15
    J[h//2:, :, 1] = 0.12
    J[h//2:, :, 2] = 0.10
    # hills silhouette
    cv2.ellipse(J, (120, 200), (150, 40), 0, 0, 360, [0.08, 0.07, 0.06], -1)
    cv2.ellipse(J, (360, 210), (130, 50), 0, 0, 360, [0.05, 0.05, 0.05], -1)
    # sun disk
    cv2.circle(J, (240, 130), 25, [1.0, 0.85, 0.4], -1)

    t = np.zeros((h, w))
    for y in range(h):
        if y < h//2:
            t[y, :] = 0.65
        else:
            f = (y - h//2) / (h//2)
            t[y, :] = 0.55 + 0.40 * f
    t = cv2.GaussianBlur(t, (0, 0), 5)
    return J, t

J_gt, t_gt = make_sunset()
A = np.array([0.90, 0.75, 0.65])   # warm haze
I = t_gt[..., None] * J_gt + (1 - t_gt[..., None]) * A[None, None, :]

J_std, t_std, A_std, _ = dehaze_standard(I)
J_sa, t_sa, A_sa, _, sky = dehaze_skyaware(I)

print("Sunset scenario:")
print(f"  input:        PSNR={psnr(I,J_gt):.2f}  SSIM={ssim_simple(I,J_gt):.3f}")
print(f"  standard DCP: PSNR={psnr(J_std,J_gt):.2f}  SSIM={ssim_simple(J_std,J_gt):.3f}")
print(f"  sky-aware:    PSNR={psnr(J_sa,J_gt):.2f}  SSIM={ssim_simple(J_sa,J_gt):.3f}")
print(f"  A_true={A}  A_std={A_std}  A_sky={A_sa}")

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
panels = [
    ("(a) Foggy sunset", I),
    ("(b) Ground truth", J_gt),
    (f"(c) Standard DCP PSNR={psnr(J_std,J_gt):.1f}", J_std),
    (f"(d) Sky-aware DCP PSNR={psnr(J_sa,J_gt):.1f}", J_sa),
]
for ax, (t, im) in zip(axes, panels):
    ax.imshow(np.clip(im,0,1)); ax.set_title(t, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{FIG}/dcp_sunset.png", dpi=150)
plt.close()

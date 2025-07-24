import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from config import sr_config as config
import matplotlib.pyplot as plt

# Path to the Set5 dataset
file = "dataset/Set5/head.png"
scale = 4  # upscale factor

# Load the trained SRCNN model
model = load_model(config.MODEL_PATH, compile=False)

# Lists to hold metric values for all images
psnr_list, ssim_list, ifc_list, nqm_list, wpsnr_list, msssim_list = [], [], [], [], [], []

border = 6  # pixels cropped from each side to account for convolutional border effects

# Weighted PSNR: uses spatial weights (Hanning window)
def wpsnr(img1, img2):
    weight = np.hanning(img1.shape[0])[:, None] * np.hanning(img1.shape[1])[None, :]
    mse = np.sum(weight * (img1 - img2) ** 2) / np.sum(weight)
    return 10 * np.log10(1.0 / mse)

# Multi-Scale SSIM: computes SSIM over multiple scales and takes a weighted average
def msssim(img1, img2, levels=5):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # weights from the original MS-SSIM paper
    mssim = []
    for _ in range(levels):
        min_side = min(img1.shape)
        win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)
        mssim.append(ssim(img1, img2, data_range=1.0, win_size=win_size, channel_axis=None))
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
    return np.sum(np.array(mssim) * np.array(weights))

# Simplified Information Fidelity Criterion (IFC): log ratio between covariance and noise variance
def ifc_simple(ref, deg):
    ref_mean = np.mean(ref)
    deg_mean = np.mean(deg)
    cov = np.mean((ref - ref_mean) * (deg - deg_mean))
    var_ref = np.var(ref)
    var_noise = np.var(ref - deg)
    return np.log10((cov ** 2) / (var_ref * var_noise + 1e-10) + 1)

# Simplified Noise Quality Measure (NQM): inverse of (1 + MSE)
def nqm_simple(ref, deg):
    mse = np.mean((ref - deg) ** 2)
    return 1 / (1 + mse)

# Loop through all images in the dataset
# for file in os.listdir(set5_dir):
    # if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
        # Read the high-resolution (ground truth) image
hr = cv2.imread(file)
hr_ycrcb = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
y = hr_ycrcb[:, :, 0]  # use only the luminance (Y) channel
h, w = y.shape

# Create low-resolution version by downscaling and then upscaling using bicubic interpolation
lr = cv2.resize(y, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

# Run SRCNN prediction
input_img = lr_up.reshape(1, h, w, 1)
pred = model.predict(input_img)
pred_y = np.clip(pred[0].squeeze(), 0, 1)

# Crop images to avoid border effects
y_cropped = y[border:h-border, border:w-border]
pred_cropped = pred_y[0:y_cropped.shape[0], 0:y_cropped.shape[1]]


# Compute quality metrics
psnr_val = psnr(y_cropped, pred_cropped, data_range=1.0)
ssim_val = ssim(y_cropped, pred_cropped, data_range=1.0, channel_axis=-1)
wpsnr_val = wpsnr(y_cropped, pred_cropped)
msssim_val = msssim(y_cropped, pred_cropped)
ifc_val = ifc_simple(y_cropped, pred_cropped)
nqm_val = nqm_simple(y_cropped, pred_cropped)

# Append metrics to lists
psnr_list.append(psnr_val)
ssim_list.append(ssim_val)
wpsnr_list.append(wpsnr_val)
msssim_list.append(msssim_val)
ifc_list.append(ifc_val)
nqm_list.append(nqm_val)

# Print per-image metrics
print(f"{file}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}, WPSNR={wpsnr_val:.2f} dB, "
        f"MSSIM={msssim_val:.4f}, IFC={ifc_val:.4f}, NQM={nqm_val:.4f}")
        # break;

# Print average metrics across all images
print(f"\nAverage:")
print(f"PSNR={np.mean(psnr_list):.2f} dB")
print(f"SSIM={np.mean(ssim_list):.4f}")
print(f"WPSNR={np.mean(wpsnr_list):.2f} dB")
print(f"MSSIM={np.mean(msssim_list):.4f}")
print(f"IFC={np.mean(ifc_list):.4f}")
print(f"NQM={np.mean(nqm_list):.4f}")

# --- Crop tất cả ảnh theo kích thước của SRCNN output ---
h_pred, w_pred = pred_y.shape
hr_crop = hr_ycrcb[:h_pred, :w_pred, :]
lr_up_crop = lr_up[:h_pred, :w_pred]

# LR image (bicubic upscaled)
lr_img = np.zeros_like(hr_crop)
lr_img[:, :, 0] = lr_up_crop
lr_img[:, :, 1:] = hr_crop[:, :, 1:]
lr_img = cv2.cvtColor((lr_img * 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# SRCNN output image
sr_img = np.zeros_like(hr_crop)
sr_img[:, :, 0] = pred_y
sr_img[:, :, 1:] = hr_crop[:, :, 1:]
sr_img = cv2.cvtColor((sr_img * 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Ground truth HR image
hr_img = cv2.cvtColor((hr_crop * 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# --- Hiển thị 3 ảnh ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(lr_img)
plt.title("Input (LR Bicubic)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sr_img)
plt.title(f"SRCNN Output\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hr_img)
plt.title("Ground Truth (HR)")
plt.axis("off")

plt.tight_layout()
plt.show()
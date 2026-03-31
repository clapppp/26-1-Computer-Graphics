import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

def histogram_equlize(channel):
    hist = np.bincount(channel.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero_cdf = cdf[cdf > 0]

    if len(nonzero_cdf) == 0:
        return channel.copy()

    cdf_min = nonzero_cdf[0]

    if channel.size == cdf_min:
        return channel.copy()

    lut = (cdf - cdf_min) / (channel.size - cdf_min) * 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[channel]


def enhance_image(input_path, output_path):
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    result_gray = histogram_equlize(img_gray)

    ok = cv2.imwrite(output_path, result_gray)
    if not ok:
        raise ValueError(f"이미지 저장 실패: {output_path}")

    return img_gray, result_gray


def compute_metrics(ref_gray, test_gray):
    ref_f = ref_gray.astype(np.float32)
    test_f = test_gray.astype(np.float32)

    mse = mean_squared_error(ref_f, test_f)
    psnr = peak_signal_noise_ratio(ref_f, test_f, data_range=255)
    ssim = structural_similarity(ref_f, test_f, data_range=255)

    return mse, psnr, ssim


def plot_result(input_img, output_img, save_path):
    mse, psnr, ssim = compute_metrics(input_img, output_img)

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(input_img, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(output_img, cmap="gray")
    plt.title("Equalized")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.plot(np.bincount(input_img.ravel(), minlength=256), color="black")
    plt.title("Input Hist")
    plt.xlim([0, 256])

    plt.subplot(1, 5, 4)
    plt.plot(np.bincount(output_img.ravel(), minlength=256), color="black")
    plt.title("Output Hist")
    plt.xlim([0, 256])

    plt.subplot(1, 5, 5)
    plt.axis("off")
    plt.text(
        0.0, 0.5,
        f"MSE  : {mse:.4f}\nPSNR : {psnr:.4f}\nSSIM : {ssim:.4f}",
        fontsize=12,
        va="center"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    input_path = "inputs/grayscale.jpg"
    output_path = "outputs/output_gray_he.jpg"
    figure_path = "outputs/grayscale_hist_equalization.png"

    input_img, output_img = enhance_image(input_path, output_path)
    plot_result(input_img, output_img, figure_path)

main()
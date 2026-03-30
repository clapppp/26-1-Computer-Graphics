import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

def simple_ahe(channel, tile_size):
    h, w = channel.shape
    tile_h, tile_w = tile_size
    result = np.zeros_like(channel)

    for y0 in range(0, h, tile_h):
        for x0 in range(0, w, tile_w):
            y1 = min(y0 + tile_h, h)
            x1 = min(x0 + tile_w, w)

            tile = channel[y0:y1, x0:x1]
            result[y0:y1, x0:x1] = cv2.equalizeHist(tile)

    return result


def enhance_rgb_channels(input_path, output_path, method):
    img_bgr = cv2.imread(input_path)

    if img_bgr is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)

    if method == "he":
        r2 = cv2.equalizeHist(r)
        g2 = cv2.equalizeHist(g)
        b2 = cv2.equalizeHist(b)

    elif method == "ahe":
        r2 = simple_ahe(r, tile_size=(64, 64))
        g2 = simple_ahe(g, tile_size=(64, 64))
        b2 = simple_ahe(b, tile_size=(64, 64))

    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        r2 = clahe.apply(r)
        g2 = clahe.apply(g)
        b2 = clahe.apply(b)

    else:
        raise ValueError("method는 'he', 'ahe', 'clahe' 중 하나여야 합니다.")

    result_rgb = cv2.merge((r2, g2, b2))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(output_path, result_bgr)
    if not ok:
        raise ValueError(f"이미지 저장 실패: {output_path}")

    return result_bgr


def compute_metrics(ref_rgb, test_rgb):
    ref_f = ref_rgb.astype(np.float32)
    test_f = test_rgb.astype(np.float32)

    mse = mean_squared_error(ref_f, test_f)
    psnr = peak_signal_noise_ratio(ref_f, test_f, data_range=255)
    ssim = structural_similarity(ref_f, test_f, data_range=255, channel_axis=2)

    return mse, psnr, ssim


def plot_images_histograms_and_metrics():
    image_paths = [
        ("Input", "input.jpg"),
        ("RGB HE", "outputs/output_rgb_he.jpg"),
        ("RGB AHE", "outputs/output_rgb_ahe.jpg"),
        ("RGB CLAHE", "outputs/output_rgb_clahe.jpg"),
    ]

    ref_bgr = cv2.imread("input.jpg")
    if ref_bgr is None:
        raise ValueError("기준 이미지 input.jpg를 불러올 수 없습니다.")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(4, 5, figsize=(24, 16))
    fig.suptitle("Image + RGB Histograms + Metrics", fontsize=18)

    colors = ["r", "g", "b"]
    channel_names = ["R", "G", "B"]

    for row, (title, path) in enumerate(image_paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ax_img = axes[row, 0]
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"{title} - Image")
        ax_img.axis("off")

        for ch in range(3):
            hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256])

            ax = axes[row, ch + 1]
            ax.plot(hist, color=colors[ch], linewidth=1.5)
            ax.set_title(f"{title} - {channel_names[ch]}")
            ax.set_xlim([0, 256])
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)

        ax_text = axes[row, 4]
        ax_text.axis("off")

        if title == "Input":
            metric_text = (
                "Reference Image\n\n"
                "MSE  : 0.0000\n"
                "PSNR : inf\n"
                "SSIM : 1.0000"
            )
        else:
            mse, psnr, ssim = compute_metrics(ref_rgb, img_rgb)
            metric_text = (
                f"{title} vs Input\n\n"
                f"MSE  : {mse:.4f}\n"
                f"PSNR : {psnr:.4f}\n"
                f"SSIM : {ssim:.4f}"
            )

        ax_text.text(
            0.05, 0.5, metric_text,
            fontsize=13,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray")
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("outputs/image_hist_metrics.png", dpi=200)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    enhance_rgb_channels("input.jpg", "outputs/output_rgb_he.jpg", method="he")
    enhance_rgb_channels("input.jpg", "outputs/output_rgb_ahe.jpg", method="ahe")
    enhance_rgb_channels("input.jpg", "outputs/output_rgb_clahe.jpg", method="clahe")

    plot_images_histograms_and_metrics()


if __name__ == "__main__":
    main()

# python3 -m venv .venv
# ./.venv/bin/python -m pip install opencv-python numpy matplotlib scikit-image
# ./.venv/bin/python contrast.py
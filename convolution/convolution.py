import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv(input, se, type):
    h, w = input.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(input, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros_like(input)

    for y in range(h):
        for x in range(w):
            window = padded[y:y+kh, x:x+kw]

            if type == "dilate":
                if np.any(window[se == 1] == 255):
                    out[y, x] = 255
            elif type == "erode":
                if np.all(window[se == 1] == 255):
                    out[y, x] = 255

    return out

def plot_result(binary, binary_dilation, binary_erosion):
    _, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(binary, cmap="gray")
    ax[0].set_title("Binary")
    ax[0].axis("off")

    ax[1].imshow(binary_dilation, cmap="gray")
    ax[1].set_title("Dilation")
    ax[1].axis("off")

    ax[2].imshow(binary_erosion, cmap="gray")
    ax[2].set_title("Erosion")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/result.png", dpi=200)
    plt.show()

def main():
    binary = cv2.imread("./inputs/binary.png", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)

    binary_dilation = conv(binary, kernel, "dilate")
    binary_erosion = conv(binary, kernel, "erode")

    os.makedirs("outputs", exist_ok=True)
    
    plot_result(binary, binary_dilation, binary_erosion)

main()
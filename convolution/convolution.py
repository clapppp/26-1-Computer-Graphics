import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    binary = cv2.imread("./inputs/binary.png", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)

    binary_dilation = cv2.dilate(binary, kernel, iterations=1)
    binary_erosion = cv2.erode(binary, kernel, iterations=1)

    os.makedirs("outputs", exist_ok=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

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

main()
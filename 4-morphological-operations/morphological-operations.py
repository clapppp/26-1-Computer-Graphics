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

def plot_result(result):
        row_titles = ["Dilation / Erosion", "Opening", "Closing"]
        col_titles = ["Input", "Result 1", "Result 2"]

        _, axes = plt.subplots(3, 3, figsize=(12, 12))

        for i in range(3):
                for j in range(3):
                        ax = axes[i][j]
                        img = result[i][j]

                        if img is None:
                                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
                        else:
                                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

                        if i == 0:
                                ax.set_title(col_titles[j], fontsize=13)
                        if j == 0:
                                ax.set_ylabel(row_titles[i], fontsize=13)

                        ax.set_xticks([])
                        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig("4-morphological-operations/result.png", dpi=200)
        plt.close()

def main():
        binary = cv2.imread("4-morphological-operations/inputs/binary.png", cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)

        binary_dilation = conv(binary, kernel, "dilate")
        binary_erosion = conv(binary, kernel, "erode")

        opening_image = cv2.imread("4-morphological-operations/inputs/opening.png",cv2.IMREAD_GRAYSCALE)
        closing_image = cv2.imread("4-morphological-operations/inputs/closing.png",cv2.IMREAD_GRAYSCALE)
        _, opening_image = cv2.threshold(opening_image,127,255,cv2.THRESH_BINARY)
        _, closing_image = cv2.threshold(closing_image,127,255,cv2.THRESH_BINARY)

        se = np.ones((20, 20), dtype=np.uint8)
        opening_result = conv(conv(opening_image,se,"erode"),se,"dilate")
        closing_result = conv(conv(closing_image,se,"dilate"),se,"erode")
        result = [[binary, binary_dilation, binary_erosion],
                  [opening_image,None, opening_result],[closing_image,None, closing_result]]

        plot_result(result);

main()
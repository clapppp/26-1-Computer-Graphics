import cv2
import numpy as np
import os
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

def plot_result(opening_image, closing_image, opening_result, closing_result):
     _, ax = plt.subplots(2,2, figsize=(10,10))

     ax[0,0].imshow(opening_image, cmap="gray")
     ax[0,0].set_title("Opening input")
     
     ax[0,1].imshow(opening_result, cmap="gray")
     ax[0,1].set_title("Opening output")

     ax[1,0].imshow(closing_image, cmap="gray")
     ax[1,0].set_title("Closing input")

     ax[1,1].imshow(closing_result, cmap="gray")
     ax[1,1].set_title("Closing output")

     plt.tight_layout()
     os.makedirs("outputs", exist_ok=True)
     plt.savefig("outputs/result.png", dpi=200)
     plt.show()

def main():
        opening_image = cv2.imread("inputs/opening.png",cv2.IMREAD_GRAYSCALE)
        closing_image = cv2.imread("inputs/closing.png",cv2.IMREAD_GRAYSCALE)
        _, opening_image = cv2.threshold(opening_image,127,255,cv2.THRESH_BINARY)
        _, closing_image = cv2.threshold(closing_image,127,255,cv2.THRESH_BINARY)

        se = np.ones((20, 20), dtype=np.uint8)
        opening_result = conv(conv(opening_image,se,"erode"),se,"dilate")
        closing_result = conv(conv(closing_image,se,"dilate"),se,"erode")

        plot_result(opening_image, closing_image, opening_result, closing_result)


main()
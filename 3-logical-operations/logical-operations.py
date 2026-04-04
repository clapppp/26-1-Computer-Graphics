import cv2
import matplotlib.pyplot as plt

def readimg():
        bi1 = cv2.imread("3-logical-operations/inputs/sub1.png")
        bi2 = cv2.imread("3-logical-operations/inputs/sub2.png")
        bi1 = cv2.cvtColor(bi1, cv2.COLOR_BGR2GRAY)
        bi2 = cv2.cvtColor(bi2, cv2.COLOR_BGR2GRAY)
        _, bi1 = cv2.threshold(bi1, 25, 255, cv2.THRESH_BINARY)
        _, bi2 = cv2.threshold(bi2, 25, 255, cv2.THRESH_BINARY)
        h1, w1 = bi1.shape[:2]
        h2, w2 = bi2.shape[:2]
        h = min(h1,h2)
        w = min(w1,w2)
        bi1 = bi1[:h,:w]
        bi2 = bi2[:h,:w]
        return bi1,bi2

def and_oper(bi1,bi2):
        return cv2.bitwise_and(bi1,bi2)
def or_oper(bi1,bi2):
        return cv2.bitwise_or(bi1,bi2)
def not_oper(bi1):
        return cv2.bitwise_not(bi1)
def plt_result(result):
        row_titles = ["AND", "OR", "NOT"]
        col_titles = ["Input 1", "Input 2", "Output"]

        _, axes = plt.subplots(3, 3, figsize=(9, 9))

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
        plt.savefig("3-logical-operations/result.png", dpi=200)
        plt.close()

def main():
        bi1, bi2 = readimg()

        and_result = and_oper(bi1,bi2)
        or_result = or_oper(bi1,bi2)
        not_result = not_oper(bi1)
        result = [[bi1,bi2,and_result],[bi1,bi2,or_result],[bi1,None,not_result]]

        plt_result(result)

main()

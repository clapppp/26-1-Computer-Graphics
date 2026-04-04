import cv2
import matplotlib.pyplot as plt
import numpy as np

def addition(input_path):
        input_add1 = cv2.imread(input_path + "add1.png")
        input_add2 = cv2.imread(input_path + "add2.png")

        h1, w1 = input_add1.shape[:2]
        h2, w2 = input_add2.shape[:2]
        target_h = min(h1, h2)
        target_w = min(w1, w2)

        b1,g1,r1 = cv2.split(input_add1[:target_h, :target_w])
        b2,g2,r2 = cv2.split(input_add2[:target_h, :target_w])

        b=cv2.add(b1,b2)
        g=cv2.add(g1,g2)
        r=cv2.add(r1,r2)

        return input_add1, input_add2, cv2.merge([b,g,r])
        
def subtraction(input_path):
        input_sub1 = cv2.imread(input_path + "sub1.png")
        input_sub2 = cv2.imread(input_path + "sub2.png")
        
        h1, w1 = input_sub1.shape[:2]
        h2, w2 = input_sub2.shape[:2]
        target_h = min(h1, h2)
        target_w = min(w1, w2)

        b1,g1,r1 = cv2.split(input_sub1[:target_h, :target_w])
        b2,g2,r2 = cv2.split(input_sub2[:target_h, :target_w])

        b=cv2.subtract(b1,b2)
        g=cv2.subtract(g1,g2)
        r=cv2.subtract(r1,r2)

        return input_sub1, input_sub2, cv2.merge([b,g,r])

def multiplication(input_path):
        input_mul = cv2.imread(input_path + "mul.png")

        b,g,r = cv2.split(input_mul)
        scale = 2
        b=cv2.multiply(b,scale)
        g=cv2.multiply(g,scale)
        r=cv2.multiply(r,scale)

        return input_mul, cv2.merge([b,g,r])

def division(input_path):
        input_div1 = cv2.imread(input_path + "div1.png")
        input_div2 = cv2.imread(input_path + "div2.png")
        
        h1, w1 = input_div1.shape[:2]
        h2, w2 = input_div2.shape[:2]
        target_h = min(h1, h2)
        target_w = min(w1, w2)

        b1,g1,r1 = cv2.split(input_div1[:target_h, :target_w])
        b2,g2,r2 = cv2.split(input_div2[:target_h, :target_w])
        
        b2 = np.where(b2 == 0, 1, b2).astype(np.float32)
        g2 = np.where(g2 == 0, 1, g2).astype(np.float32)
        r2 = np.where(r2 == 0, 1, r2).astype(np.float32)

        b = cv2.divide(b1.astype(np.float32), b2, scale=255.0)
        g = cv2.divide(g1.astype(np.float32), g2, scale=255.0)
        r = cv2.divide(r1.astype(np.float32), r2, scale=255.0)

        b = np.clip(b, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        r = np.clip(r, 0, 255).astype(np.uint8)

        return input_div1, input_div2, cv2.merge([b,g,r])

def plot_result(output, figure_path):
        row_titles = ["Addition", "Subtraction", "Multiplication", "Division"]
        col_titles = ["Input 1", "Input 2", "Output"]

        _, axes = plt.subplots(4, 3, figsize=(14, 16))

        for i in range(4):
                for j in range(3):
                        ax = axes[i][j]
                        img = output[i][j]
                        if img is None:
                                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16)
                        else:
                                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

                        if i == 0:
                                ax.set_title(col_titles[j], fontsize=13)
                        if j == 0:
                                ax.set_ylabel(row_titles[i], fontsize=13)

                        ax.set_xticks([])
                        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()

def main():
        input_path = "2-arithmetic-operations/inputs/"
        figure_path = "2-arithmetic-operations/result.png"

        input_add1, input_add2, output_add = addition(input_path)
        input_sub1, input_sub2, output_sub = subtraction(input_path)
        input_mul, output_mul = multiplication(input_path)
        input_div1, input_div2, output_div = division(input_path)

        output = [
                [input_add1, input_add2, output_add],
                [input_sub1, input_sub2, output_sub],
                [input_mul,  None,       output_mul],
                [input_div1, input_div2, output_div]
        ]        

        plot_result(output ,figure_path)

main()
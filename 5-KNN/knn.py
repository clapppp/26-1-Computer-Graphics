import pandas
from PIL import Image
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

df = pandas.read_csv("./dataset/trainLabels.csv")
labels = df.set_index("id")["label"].to_dict()  #{1:frog, 2:car ..}
dataset = []
for i in range(1,50001):
        if i%10000==0:print(f"  {i}/50000 loaded")
        img = Image.open(f"./dataset/train/{i}.png")
        pixel = np.array(img).flatten()
        row = [pixel, labels[i]]
        dataset.append(row)
testset  = dataset[:100]
trainset = dataset[10000:]
print(f"Complete load, testset:{len(testset)}, trainset:{len(trainset)}")

x_test  = [row[0] for row in testset]
y_test  = [row[1] for row in testset]
x_train = [row[0] for row in trainset]
y_train = [row[1] for row in trainset]
classes = sorted(list(set(y_train))) + ["none"]
class_to_idx = {c:i for i,c in enumerate(classes)}

def knn(x, k, metrics):
        klist = []
        for i, xt in enumerate(x_train):
                dist = metrics(x,xt)
                t = [dist,i]
                klist.append(t)
        klist = sorted(klist)[:k]
        labels = [y_train[i] for _,i in klist]
        most = Counter(labels).most_common(2)
        if len(most) > 1 and most[0][1] == most[1][1]:return "none"
        return most[0][0]

def L1(x, xt):
        return np.sum(np.abs(x-xt))
def L2(x, xt):
        return np.sqrt(np.sum((x-xt) ** 2))

def save_result(accuracy, confusion, k, metric):
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(confusion, cmap='Blues')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Metric={metric}, K={k}, Accuracy={accuracy:.4f}')

        thresh = confusion.max() / 2
        for i in range(len(classes)):
                for j in range(len(classes)):
                        ax.text(j, i, str(confusion[i][j]), ha='center', va='center',
                                color='white' if confusion[i][j] > thresh else 'black')

        plt.tight_layout()
        plt.savefig(f'./results/{metric}_K{k}.png', dpi=150, bbox_inches='tight')
        print(f"  K:{k}, metric:{metric}, {accuracy}")
        plt.close()

K = [1,3,5,7,9]
for k in K:
        print(f"K:{k}, metric:L1 - started")
        correct = 0
        confusion = np.zeros((len(classes), len(classes)), dtype=int)# 11X11
        for i, (x, y) in enumerate(zip(x_test, y_test)):
                if i%10 == 0:print(f"  {i}/{len(x_test)}")
                yt = knn(x,k,L1)
                confusion[class_to_idx[y]][class_to_idx[yt]]+=1
                if y==yt : correct+=1
        save_result(correct/len(y_test), confusion, k, "L1")

for k in K:
        print(f"K:{k}, metric:L2 - started")
        correct = 0
        confusion = np.zeros((len(classes), len(classes)), dtype=int)# 11X11
        for i, (x, y) in enumerate(zip(x_test, y_test)):
                if i%10 == 0:print(f"  {i}/{len(x_test)}")
                yt = knn(x,k,L2)
                confusion[class_to_idx[y]][class_to_idx[yt]]+=1
                if y==yt : correct+=1
        save_result(correct/len(y_test), confusion, k, "L2")
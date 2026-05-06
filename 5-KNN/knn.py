import pandas
from PIL import Image
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

df = pandas.read_csv("./dataset/trainLabels.csv")
labels = df.set_index("id")["label"].to_dict()  #{1:frog, 2:car ..}
dataset = []
for i in range(1,10001):
        if i%10000==0:print(f"  {i}/50000 loaded")
        img = Image.open(f"./dataset/train/{i}.png")
        pixel = np.array(img).flatten()
        row = [pixel, labels[i]]
        dataset.append(row)

five_folds = []
for i in range(6): five_folds.append(dataset[i*1000:i*1000+1000])
print("Complete load")

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
def save_result(accuracy, confusion, k, metric, fold):
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
        plt.savefig(f'./results/confusion_matrix/{metric}_K{k}_{fold}fold.png', dpi=150, bbox_inches='tight')
        print(f"  K:{k}, metric:{metric}, {accuracy}")
        plt.close()

result_acc = np.zeros((2,5,6))

for i in range(6): 
        print(f"{i} fold is running")
        testset = five_folds[i]
        trainset = []
        for j in range(6): 
                if j != i: trainset += five_folds[j]
        x_test  = [row[0] for row in testset]
        y_test  = [row[1] for row in testset]
        x_train = [row[0] for row in trainset]
        y_train = [row[1] for row in trainset]
        classes = sorted(list(set(y_train))) + ["none"]
        class_to_idx = {c:j for j,c in enumerate(classes)}
        K = [1,3,5,7,9]
        for k in K:
                print(f"K:{k}, metric:L1 - started")
                correct = 0
                confusion = np.zeros((len(classes), len(classes)), dtype=int)# 11X11
                for j, (x, y) in enumerate(zip(x_test, y_test)):
                        if j%100 == 0:print(f"  {j}/{len(x_test)}")
                        yt = knn(x,k,L1)
                        confusion[class_to_idx[y]][class_to_idx[yt]]+=1
                        if y==yt : correct+=1
                accuracy = correct/len(y_test) * 100
                result_acc[0,K.index(k),i] = accuracy
                save_result(accuracy, confusion, k, "L1",i)

        for k in K:
                print(f"K:{k}, metric:L2 - started")
                correct = 0
                confusion = np.zeros((len(classes), len(classes)), dtype=int)# 11X11
                for j, (x, y) in enumerate(zip(x_test, y_test)):
                        if j%100 == 0:print(f"  {j}/{len(x_test)}")
                        yt = knn(x,k,L2)
                        confusion[class_to_idx[y]][class_to_idx[yt]]+=1
                        if y==yt : correct+=1
                accuracy = correct/len(y_test) * 100
                result_acc[1,K.index(k),i] = accuracy
                save_result(accuracy, confusion, k, "L2",i)

# ── Accuracy 시각화 ──────────────────────────────────────────
K = [1, 3, 5, 7, 9]
metrics_name = ["L1", "L2"]
colors = ["steelblue", "tomato"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("KNN Accuracy (6-Fold Cross Validation)", fontsize=14, fontweight="bold")

for m_idx, (ax, metric) in enumerate(zip(axes, metrics_name)):
    fold_accs  = result_acc[m_idx]          # shape: (5, 6) → [k_idx, fold]
    mean_accs  = fold_accs.mean(axis=1)     # k별 평균
    std_accs   = fold_accs.std(axis=1)      # k별 표준편차

    # fold별 선
    for f in range(6):
        ax.plot(K, fold_accs[:, f], color=colors[m_idx],
                alpha=0.25, linewidth=1, linestyle="--")

    # 평균 선 (굵게)
    ax.plot(K, mean_accs, color=colors[m_idx],
            linewidth=2.5, marker="o", label="Mean")

    # 표준편차 음영
    ax.fill_between(K,
                    mean_accs - std_accs,
                    mean_accs + std_accs,
                    color=colors[m_idx], alpha=0.15, label="±1 std")

    # 평균값 텍스트
    for k_val, acc in zip(K, mean_accs):
        ax.annotate(f"{acc:.2f}%", (k_val, acc),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)

    ax.set_title(f"Metric: {metric}", fontsize=12)
    ax.set_xlabel("K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(K)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("./results/accuracy_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Accuracy plot saved.")
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = []
grams = [1, 2, 3, 4, 5]
ks = np.array(range(0, 20)) / 10
for gram in grams:
    for k in ks:
        with open(f"logs/{k}_{gram}.log", 'r') as fin:
            lines = [line for line in fin]
        acc = float(lines[-1].split("Accuracy: ")[-1][:-2])
        data.append(acc)
        # print(acc)
data = np.array(data).reshape([len(grams), len(ks)])
img = sns.heatmap(data, xticklabels=ks, yticklabels=grams, square=True, cbar_kws={"shrink": 0.26}, vmax=0.75, vmin=0.6, linewidths=1)
img.set_xlabel('Add-k smooth')
img.set_ylabel('N-grams')
plt.savefig("result.png", dpi=100, bbox_inches='tight')
print(data.max())
print(data.argmax(axis=1))
print(grams[data.max(axis=1).argmax()], ks[data.max(axis=0).argmax()])

k = 0.0
for gram in grams:
    with open(f"logs/{k}_{gram}.log", 'r') as fin:
        lines = [line for line in fin]
    a = lines[-1].split("Accuracy: ")[-1][:-2]
    f = lines[-1].split(", F1: ")[-1].split(', ')[0]
    r = lines[-1].split(", Recall: ")[-1].split(', ')[0]
    p = lines[-1].split("Precision: ")[-1].split(', ')[0]
    t = ['', str(gram), p, r, f, a, '']
    print(' | '.join(t))

gram = 1
for k in ks[::2]:
    with open(f"logs/{k}_{gram}.log", 'r') as fin:
        lines = [line for line in fin]
    a = lines[-1].split("Accuracy: ")[-1][:-2]
    f = lines[-1].split(", F1: ")[-1].split(', ')[0]
    r = lines[-1].split(", Recall: ")[-1].split(', ')[0]
    p = lines[-1].split("Precision: ")[-1].split(', ')[0]
    t = ['', str(k), p, r, f, a, '']
    print(' | '.join(t))
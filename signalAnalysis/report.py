from Dataset import Dataset

import pandas as pd
import numpy as np
import os
import glob

from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import imageio


def evaluate(prediction, labels):
    """
        Function to evaluate the results of a clustering method suing Point Biserial and AUCC for clustering.
    """

    from scipy.spatial.distance import pdist
    distance_matrix = pdist(prediction, "euclidean")

    noiseSize = 0
    for value in labels:
        if value == 0:
            noiseSize += 1

    penalty = (len(labels) - noiseSize) / len(labels)

    if noiseSize == len(labels):
        return np.nan, np.nan, noiseSize, penalty

    dm = squareform(distance_matrix)
    x = []
    yPointBiserial = []
    yAucc = []

    for i in range(len(labels) - 1):
        if labels[i] == 0:
            continue

        for j in range(i, len(labels)):
            if labels[j] == 0:
                continue

            yPointBiserial.append(dm[i, j])
            yAucc.append(1/(1+dm[i, j]))

            if labels[i] == labels[j]:
                x.append(1)
            else:
                x.append(0)

    # Compute internal validity index (point biserial)
    pb, pv = pointbiserialr(x, yPointBiserial)

    # Compute area under the curve
    aucc = roc_auc_score(x, yAucc)

    silhouette = silhouette_score(prediction, labels)

    return penalty*pb, penalty*aucc, penalty*silhouette, noiseSize, penalty


os.makedirs("./report", exist_ok=True)
os.makedirs("./conrado", exist_ok=True)

prediction = np.load("results/prediction.npy")
results = pd.read_csv("results/results.csv", index_col=0)

# print('Reporting')
# report = pd.DataFrame(columns=["Method", "PB", "AUCC", "Silhouette", "Noise", "Penalty"])
# for method in results.columns:
#     labels = results[method].values
#     report.loc[len(report)] = [method] + list(evaluate(prediction, labels))
# report.to_csv(f"report/report.csv")

df = pd.read_csv('report/report.csv', index_col=0)[['Method', 'AUCC', 'Silhouette']]
max1, max2, max3 = ('', 0), ('', 0), ('', 0)
for method, aucc, silhouette in zip(df.values[:, 0], df.values[:, 1], df.values[:, 2]):
    for alpha in [0.1, 0.5, 0.9]:
        score = alpha*aucc + (1-alpha)*silhouette
        if alpha == 0.1 and max1[1] < score:
            max1 = (method, score)
        if alpha == 0.5 and max2[1] < score:
            max2 = (method, score)
        if alpha == 0.9 and max3[1] < score:
            max3 = (method, score)
print(max1, max2, max3)

print('Plotting')
tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50, )
tsne_2d = tsne_results.fit_transform(prediction)
tsne_2d = pd.DataFrame({'x': tsne_2d[:, 0], 'y': tsne_2d[:, 1]})

for method in [max1[0], max2[0], max3[0]]:
    tsne_2d['labels'] = results[method].values
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(data=tsne_2d, x='x', y='y', c='labels', cmap='jet', alpha=0.5)
    for label in set(tsne_2d['labels'].values):
        x = tsne_2d.where(tsne_2d['labels'] == label).dropna()['x'].median()
        y = tsne_2d.where(tsne_2d['labels'] == label).dropna()['y'].median()
        center = (x, y)
        ax.annotate(str(label), xy=center, size=10, bbox=dict(boxstyle="circle", facecolor='grey'))
    plt.savefig(f'report/{method}.png', dpi=100)
    plt.close(fig)

print('Conrado')
dataset = Dataset()
dataset.dataset('signals')
for method in [max1[0], max2[0], max3[0]]:
    labels = results[method].values
    for cluster in set(labels):
        data = dataset.original_data.iloc[[True if label == cluster else False for label in labels]]
        data = data.reset_index(drop=True).values
        side_size = 6
        fig, ax = plt.subplots(side_size, side_size, figsize=(30, 30))
        rand_index = np.floor(np.linspace(0, data.shape[0]-1, side_size**2))
        for i in range(side_size**2):
            ax[int(np.floor(i / side_size)), int(i % side_size)].plot(data[int(rand_index[i])])
        plt.savefig(f'conrado/{method}_label_{cluster}.png', dpi=100)
        plt.close()

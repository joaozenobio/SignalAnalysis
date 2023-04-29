import pandas as pd
import numpy as np
import os
import glob

from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, silhouette_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import imageio

from Dataset import Dataset


def evaluate(prediction, labels):
    """
        Function to evaluate the results of a clustering method using Point Biserial,
        AUCC and silhouette for clustering.
    """

    from scipy.spatial.distance import pdist
    distance_matrix = pdist(prediction, "euclidean")

    noiseSize = 0
    for value in labels:
        if value == 0:
            noiseSize += 1

    penalty = (len(labels) - noiseSize) / len(labels)

    if noiseSize == len(labels):
        return np.nan, np.nan, np.nan, noiseSize, penalty

    dm = squareform(distance_matrix)
    x = []
    # yPointBiserial = []
    yAucc = []

    for i in range(len(labels) - 1):
        if labels[i] == 0:
            continue

        for j in range(i, len(labels)):
            if labels[j] == 0:
                continue

            # yPointBiserial.append(dm[i, j])
            yAucc.append(1/(1+dm[i, j]))

            if labels[i] == labels[j]:
                x.append(1)
            else:
                x.append(0)

    # Compute internal validity index (point biserial)
    # pb, pv = pointbiserialr(x, yPointBiserial)

    # Compute area under the curve
    aucc = 0
    if len(set(x)) > 1:
        aucc = roc_auc_score(x, yAucc)

    silhouette = silhouette_score(prediction, labels)

    return 0., penalty*aucc, penalty*silhouette, noiseSize, penalty


def report():
    os.makedirs("./report", exist_ok=True)
    os.makedirs("./cluster_examples", exist_ok=True)

    prediction = np.load("results/prediction.npy")
    results = pd.read_csv("results/results.csv", index_col=0)
    original_data = pd.read_csv("original_data.csv", index_col=0)

    print('Reporting')
    report = pd.DataFrame(columns=["Method", "PB", "AUCC", "Silhouette", "Noise", "Penalty"])
    for method in results.columns:
        labels = results[method].values
        report.loc[len(report)] = [method] + list(evaluate(prediction, labels))
    report.to_csv(f"report/report.csv")

    df = pd.read_csv('report/report.csv', index_col=0)[['Method', 'PB', 'AUCC', 'Silhouette']]

    method_avg = ''
    avg = 0
    for method, pb, aucc, silhouette in zip(df.values[:, 0], df.values[:, 1], df.values[:, 2], df.values[:, 3]):
        avg_temp = (abs(pb) * aucc * silhouette) / 3
        if avg_temp > avg:
            avg = avg_temp
            method_avg = method

    print('Plotting')
    tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50)
    tsne_2d = tsne_results.fit_transform(prediction)
    tsne_2d = pd.DataFrame({'x': tsne_2d[:, 0], 'y': tsne_2d[:, 1]})
    tsne_2d['label'] = results[method_avg].values
    tsne_2d['true_label'] = original_data['label'].values
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(data=tsne_2d, x='x', y='y', c='label', cmap='jet', alpha=0.5)
    for label in set(tsne_2d['label'].values):
        x = tsne_2d.where(tsne_2d['label'] == label).dropna()['x'].median()
        y = tsne_2d.where(tsne_2d['label'] == label).dropna()['y'].median()
        center = (x, y)
        ax.annotate(str(label), xy=center, size=10, bbox=dict(boxstyle="circle", facecolor='grey'))
    print(method_avg)
    for label in set(tsne_2d['true_label'].values):
        print(f'{label}:', np.histogram(
            tsne_2d.where(tsne_2d['true_label'] == label).dropna()['label'].values,
            bins=list(range(max(tsne_2d['label'].values)+2))
        ))
    plt.savefig(f'report/{method_avg}.png', dpi=100)
    plt.close(fig)

    print('Plotting examples from clusters')
    labels = results[method_avg].values
    for cluster in set(labels):
        data = original_data.iloc[[True if label == cluster else False for label in labels]]
        data = data.reset_index(drop=True).values
        side_size = 6
        fig, ax = plt.subplots(side_size, side_size, figsize=(30, 30))
        rand_index = np.floor(np.linspace(0, data.shape[0]-1, side_size**2))
        for i in range(side_size**2):
            ax[int(np.floor(i / side_size)), int(i % side_size)].plot(
                data[int(rand_index[i])]
            )
            ax[int(np.floor(i / side_size)), int(i % side_size)].set_title(
                original_data.index.values.tolist()[int(rand_index[i])]
            )
        plt.savefig(f'cluster_examples/{method_avg}_label_{cluster}.png', dpi=100)
        plt.close()


os.makedirs("./old/means", exist_ok=True)
results = pd.read_csv("./old/results_old/results_new_FOSC_96.csv", index_col=0)
lastObjects = pd.read_csv("./old/results_old/lastObjects_FOSC_96.csv", index_col=0)
dataset = Dataset()
dataset.dataset('./old/signals_old')
original_data = pd.DataFrame(dataset.data.squeeze())
labels = results["0"].values
prediction = np.load("./old/results_old/prediction.npy")

# print(evaluate(prediction, labels))

for cluster in lastObjects.index:
    fig, ax = plt.subplots(figsize=(15, 15))
    data = original_data.iloc[lastObjects.loc[cluster][0]]
    data = data.reset_index(drop=True).values
    ax.plot(data)
    ax.set(title=f'Cluster_{cluster}_example')
    plt.savefig(f'./old/means/FOSC_96_avarage_last_object_{cluster}.jpg', dpi=200)
    plt.close()

tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50)
tsne_2d = tsne_results.fit_transform(prediction)
tsne_2d = pd.DataFrame({'x': tsne_2d[:, 0], 'y': tsne_2d[:, 1]})
tsne_2d['label'] = labels
fig, ax = plt.subplots(figsize=(15, 15))
scatter = ax.scatter(data=tsne_2d, x='x', y='y', c='label', cmap='jet', alpha=0.5)
for label in set(tsne_2d['label'].values):
    if label != 0:
        x = tsne_2d.where(tsne_2d['label'] == label).dropna()['x'].median()
        y = tsne_2d.where(tsne_2d['label'] == label).dropna()['y'].median()
        center = (x, y)
        ax.annotate(str(label), xy=center, size=10, bbox=dict(boxstyle="circle", facecolor='grey'))
plt.savefig(f'./old/means/FOSC_96_avarage_TSNE2D', dpi=100)
plt.close(fig)

# i = 0
# fig, ax = plt.subplots(5, 3, figsize=(15, 20), sharex=True, sharey=True)
# ax_list = ax.reshape(-1)
# plt.locator_params(nbins=10)
# for cluster in set(labels):
#     data = original_data.iloc[[True if label == cluster else False for label in labels]]
#     data = data.reset_index(drop=True).values
#     for signal in data:
#         ax_list[i].plot(signal, alpha=0.02)
#     ax_list[i].set(title=f'Cluster{i}')
#     i += 1
# plt.savefig('./old/means/FOSC_96_avarage_all.png', dpi=100)
# plt.close()

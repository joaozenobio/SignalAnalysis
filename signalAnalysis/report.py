from Autoencoder import Autoencoder
from Dataset import Dataset

import pandas as pd
import numpy as np
import os
import glob

from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, silhouette_score

import matplotlib as mpl
import matplotlib.pyplot as plt
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

    return penalty*pb, penalty*aucc, silhouette, noiseSize, penalty


os.makedirs("./report", exist_ok=True)
os.makedirs("./gif", exist_ok=True)
os.makedirs("./gifs", exist_ok=True)

prediction = np.load("results/prediction.npy")
results = pd.read_csv("results/results.csv", index_col=0)

processed_signal = Dataset.process_signal("result_aJdKrfM.csv")

model = Autoencoder()
model.load("model4")
observation_prediction = model.predict(processed_signal)

print('Creating report')

report = pd.DataFrame(columns=["Method", "PB", "AUCC", "Silhouette", "Noise", "Penalty"])

for method in results.columns:
    labels = results[method].values
    report.loc[len(report)] = [method] + list(evaluate(prediction, labels))

report.to_csv(f"report/report.csv")


# TODO: AKS JADSON HOW TO DECIDE THE BEST METHOD FROM THE METRICS


print('Creating gif')

tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50)
tsne_2d = tsne_results.fit_transform(np.vstack([prediction, observation_prediction]))
prediction_2d = tsne_2d[:-len(observation_prediction)]
observation_prediction_2d = tsne_2d[-len(observation_prediction):]

method = "HDBSCAN_5"
labels = results[method].values

fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(prediction_2d[:, 0], prediction_2d[:, 1], c=labels, cmap=mpl.colormaps["hsv"])
for i in range(observation_prediction_2d.shape[0]-1):
    x = observation_prediction_2d[i, 0]
    y = observation_prediction_2d[i, 1]
    x2 = observation_prediction_2d[i+1, 0]
    y2 = observation_prediction_2d[i+1, 1]
    ax.quiver(x, y, x2-x, y2-y, scale_units='xy', angles='xy', scale=1)
    plt.savefig(f'gif/{method}_{i}.png', dpi=100)
plt.close(fig)

with imageio.get_writer(f"gifs/{method}.gif", mode="I", duration=1) as writer:
    for filename in sorted(glob.glob("gif/*.png")):
        image = imageio.imread(filename)
        writer.append_data(image)

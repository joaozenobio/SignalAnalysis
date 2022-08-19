from scipy.cluster.hierarchy import linkage, dendrogram
from FOSC import FOSC
from Dataset import Dataset
from Autoencoder import Autoencoder
import hdbscan
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
import os
import sys
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# TODO:
#  Report
#  for + Reachability plot OPTICS
#  for + Dendrogram HDBSCAN
#  for + Dendrogram FOSC
#  Calculate metrics
#  Print example "result_..." in TSNE 2D with best metrics
#  Train YOLO with new images

sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs("./results", exist_ok=True)

dataset = Dataset("signals")

model = Autoencoder(dataset.data)
# model.fit(dataset.data)
# model.save("model2")
model.load("model2")
prediction = model.predict(dataset.data)

tsne_results = TSNE(n_components=2, learning_rate='auto', init='random')
TSNE_2D = tsne_results.fit_transform(prediction)
TSNE_2D = pd.DataFrame(TSNE_2D)
TSNE_2D["label"] = list(range(prediction.shape[0]))

listOfMClSize = [5, 10, 15, 20, 25, 30]
for m in listOfMClSize:
    hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=m)
    hdbscan_results.fit(prediction)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    hdbscan_results.condensed_tree_.plot()
    plt.savefig(f'HDBSCAN_{m}.png', dpi=100)

for m in listOfMClSize:
    optics_results = OPTICS(min_cluster_size=m).fit(prediction)
    reachability = optics_results.reachability_[optics_results.ordering_]
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.plot(reachability)
    plt.savefig(f'OPTICS_{m}.png', dpi=100)

# methodsLinkage = ["single", "average", "ward", "complete", "weighted"]
#
# for m in listOfMClSize:
#     for lm in methodsLinkage:
#         titlePlot = lm + " and mClSize = " + str(m)
#         Z = linkage(prediction, method=lm, metric="euclidean")
#         foscFramework = FOSC(Z, mClSize=m)
#         infiniteStability = foscFramework.propagateTree()
#         partition = foscFramework.findProminentClusters(1, infiniteStability)
#         TSNE_2D[f"FOSC_{m}_{lm}"] = partition
#


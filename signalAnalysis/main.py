from FOSC import FOSC
from Dataset import Dataset
from Autoencoder import Autoencoder
import hdbscan
from sklearn.cluster import OPTICS

import os
import sys
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# TODO:
#   Report
#   REPTILERECON -> Train YOLO with new images
#   Documentation


os.makedirs("./results", exist_ok=True)

dataset = Dataset()
dataset.dataset("signals")

# !!! save() writes last model in the same folder !!!
# model = Autoencoder(dataset.data)
# model.fit(dataset.data)
# model.save("model")

model = Autoencoder()
model.load("model")
prediction = model.predict(dataset.data)
np.save("results/prediction.npy", prediction)

df = pd.DataFrame()

listOfMClSize = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
for m in listOfMClSize:
    print(f"listOfMClSize = {m}")

    print("HDBSCAN")
    hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=m)
    hdbscan_results.fit(prediction)

    if -1 in hdbscan_results.labels_:
        hdbscan_results.labels_ = np.array(hdbscan_results.labels_)
        hdbscan_results.labels_ += 1

    # fig, ax = plt.subplots()
    # fig.set_size_inches(15, 15)
    # hdbscan_results.condensed_tree_.plot(log_size=True)
    # plt.savefig(f'results/HDBSCAN_{m}.png', dpi=100)
    # plt.close(fig)

    df[f"HDBSCAN_{m}"] = list(hdbscan_results.labels_)

    print("OPTICS")
    optics_results = OPTICS(min_cluster_size=m).fit(prediction)
    # reachability = optics_results.reachability_[optics_results.ordering_]

    if -1 in optics_results.labels_:
        optics_results.labels_ = np.array(optics_results.labels_)
        optics_results.labels_ += 1

    # fig, ax = plt.subplots()
    # fig.set_size_inches(15, 15)
    # ax.plot(reachability)
    # plt.savefig(f'results/OPTICS_{m}.png', dpi=100)
    # plt.close(fig)

    df[f"OPTICS_{m}"] = list(optics_results.labels_)

    methodsLinkage = ["single", "average", "ward", "complete", "weighted"]
    for lm in methodsLinkage:
        print(f"FOSC - {lm}")
        titlePlot = lm + " and mClSize = " + str(m)
        Z = linkage(prediction, method=lm, metric="euclidean")
        foscFramework = FOSC(Z, mClSize=m)
        infiniteStability = foscFramework.propagateTree()
        labels_ = foscFramework.findProminentClusters(1, infiniteStability)

        if -1 in labels_:
            labels_ = np.array(labels_)
            labels_ += 1

        # fig = plt.figure(figsize=(15, 15))
        # dn = dendrogram(Z=Z, color_threshold=None, leaf_font_size=5, leaf_rotation=45)
        # plt.savefig(f"results/FOSC_{m}_{lm}.png")
        # plt.close(fig)

        df[f"FOSC_{m}_{lm}"] = list(labels_)

    print("-------------")

df.to_csv(f"results/results.csv")

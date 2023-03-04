from FOSC import FOSC
from Dataset import Dataset
from Autoencoder import Autoencoder
import hdbscan
from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import linkage, dendrogram
import os
import sys
import pandas as pd
import numpy as np


def run(make_model=False):
    sys.setrecursionlimit(10000)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.makedirs("./results", exist_ok=True)

    dataset = Dataset()
    dataset.dataset("signals")

    if make_model:
        model = Autoencoder(dataset.data)
        model.fit(dataset.data)
        # !!! save() writes last model in the same folder !!!
        model.save("model")
    else:
        model = Autoencoder()
        model.load("model")
    prediction = model.predict(dataset.data)
    np.save("results/prediction.npy", prediction)

    df = pd.DataFrame()

    min_cluster_sizes = [5]
    min_samples = [2, 3, 5, 8, 13]
    for min_cluster_size in min_cluster_sizes:
        for min_sample in min_samples:

            print(f"HDBSCAN_{min_sample}_{min_cluster_size}")
            hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_sample)
            hdbscan_results.fit(prediction)
            if -1 in hdbscan_results.labels_:
                hdbscan_results.labels_ = np.array(hdbscan_results.labels_)
                hdbscan_results.labels_ += 1
            df[f"HDBSCAN_{min_sample}_{min_cluster_size}"] = list(hdbscan_results.labels_)

            print(f"OPTICS_{min_sample}_{min_cluster_size}")
            optics_results = OPTICS(min_cluster_size=min_cluster_size, min_samples=min_sample).fit(prediction)
            if -1 in optics_results.labels_:
                optics_results.labels_ = np.array(optics_results.labels_)
                optics_results.labels_ += 1
            df[f"OPTICS_{min_sample}_{min_cluster_size}"] = list(optics_results.labels_)

        linkage_methods = ["single", "average", "ward", "complete", "weighted"]
        for linkage_method in linkage_methods:
            print(f"FOSC_{linkage_method}_{min_cluster_size}")
            Z = linkage(prediction, method=linkage_method, metric="euclidean")
            fosc_framework = FOSC(Z, mClSize=min_cluster_size)
            infinite_stability = fosc_framework.propagateTree()
            labels_ = fosc_framework.findProminentClusters(1, infinite_stability)
            if -1 in labels_:
                labels_ = np.array(labels_)
                labels_ += 1
            df[f"FOSC_{min_cluster_size}_{linkage_method}"] = list(labels_)

    df.to_csv(f"results/results.csv")

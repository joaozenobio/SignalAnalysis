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
import matplotlib.pyplot as plt
import random

no_of_colors=200
palleteColors=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]


def plotDendrogram(Z, result, title, saveDescription=None):
    uniqueValues = np.unique(result)

    dicColors = {}
    dicColors[0] = "#000000"

    for i in range(len(uniqueValues)):
        if uniqueValues[i] != 0:
            dicColors[uniqueValues[i]] = palleteColors[i]

    colorsLeaf = {}
    for i in range(len(result)):
        colorsLeaf[i] = dicColors[result[i]]

    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    linkCols = {}
    for i, i12 in enumerate(Z[:, :2].astype(int)):
        c1, c2 = (linkCols[x] if x > len(Z) else colorsLeaf[x]
                  for x in i12)

        linkCols[i + 1 + len(Z)] = c1 if c1 == c2 else dicColors[0]

    fig = plt.figure(figsize=(15, 10))

    dn = dendrogram(Z=Z, color_threshold=None, leaf_font_size=10,
                    leaf_rotation=45, link_color_func=lambda x: linkCols[x])
    plt.title(title, fontsize=12)

    if saveDescription != None:
        plt.savefig(saveDescription)
        plt.close(fig)
        return

    plt.show()  # \


def run(model_path=None):
    sys.setrecursionlimit(10000)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # os.makedirs("./results", exist_ok=True)
    #
    # dataset = Dataset()
    # dataset.dataset("signals")
    #
    # if model_path is None:
    #     model = Autoencoder(dataset.data)
    #     model.fit(dataset.data)
    #     model.save("model")
    # else:
    #     model = Autoencoder()
    #     model.load(model_path)

    # prediction = model.predict(dataset.data)
    # np.save("results/prediction.npy", prediction)

    prediction = np.load("./old/results_old/prediction.npy")

    df = pd.DataFrame()

    # min_cluster_sizes = [5, 10, 15, 20, 25, 30]
    # min_samples = [5, 10, 15, 20, 25, 30]
    min_cluster_sizes = [96]
    for min_cluster_size in min_cluster_sizes:
        # for min_sample in min_samples:
        #
            # print(f"HDBSCAN_{min_sample}_{min_cluster_size}")
            # hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_sample)
            # hdbscan_results.fit(prediction)
            # if -1 in hdbscan_results.labels_:
            #     hdbscan_results.labels_ = np.array(hdbscan_results.labels_)
            #     hdbscan_results.labels_ += 1
            # df[f"HDBSCAN_{min_sample}_{min_cluster_size}"] = list(hdbscan_results.labels_)
            #
            # print(f"OPTICS_{min_sample}_{min_cluster_size}")
            # optics_results = OPTICS(min_cluster_size=min_cluster_size, min_samples=min_sample).fit(prediction)
            # if -1 in optics_results.labels_:
            #     optics_results.labels_ = np.array(optics_results.labels_)
            #     optics_results.labels_ += 1
            # df[f"OPTICS_{min_sample}_{min_cluster_size}"] = list(optics_results.labels_)

        # linkage_methods = ["single", "average", "ward", "complete", "weighted"]

        linkage_methods = ["average"]
        for linkage_method in linkage_methods:
            print(f"FOSC_{linkage_method}_{min_cluster_size}")
            Z = linkage(prediction, method=linkage_method, metric="euclidean")
            fosc_framework = FOSC(Z, mClSize=min_cluster_size)
            infinite_stability = fosc_framework.propagateTree()
            labels, lastObjects = fosc_framework.findProminentClusters(1, infinite_stability)
            if -1 in labels:
                labels = np.array(labels)
                labels += 1
            # df[f"FOSC_{min_cluster_size}_{linkage_method}"] = list(labels)

            titlePlot = "FOSC_96_AVG"
            saveDendrogram = "./old/results_old/dendograma.jpg"
            plotDendrogram(Z, labels, titlePlot, saveDendrogram)
            pd.DataFrame(list(labels)).to_csv(f"./old/results_old/results_new_FOSC_96.csv")
            pd.DataFrame(list(lastObjects.values()), index=list(lastObjects)).to_csv(
                f"./old/results_old/lastObjects_FOSC_96.csv"
            )
    # df.to_csv(f"results/results.csv")


run()

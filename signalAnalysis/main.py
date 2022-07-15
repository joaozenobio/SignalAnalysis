from scipy.cluster.hierarchy import linkage, dendrogram
from FOSC import FOSC
from Dataset import Dataset
from Autoencoder import Autoencoder
import hdbscan
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

palleteColors = ["#80ff72", "#8af3ff", "#7ee8fa", "#89043d", "#023c40", "#c3979f", "#797270", "#c57b57", "#07004d",
                 "#0e7c7b", "#c33149", "#f49e4c", "#2e4057", "#f2d7ee", "#bfb48f", "#a5668b", "#002500", "#720e07",
                 "#f46036", "#78290f"]


def plotPartition(x,y, result, title, saveDescription=None):
    uniqueValues = np.unique(result)

    fig = plt.figure(figsize=(15, 10))

    dicColors = {}
    dicColors[0] = "#000000"

    for  i in range(len(uniqueValues)):
        if uniqueValues[i] != 0:
            dicColors[uniqueValues[i]]= palleteColors[i]

    for k, v in dicColors.items():
        plt.scatter(x[result==k], y[result==k], color=v )

    plt.title(title, fontsize=15)

    if saveDescription != None:
        plt.savefig(saveDescription)
        plt.close(fig)
        return

    plt.show() #\


def plotDendrogram(Z, result, title, saveDescription=None):
    uniqueValues = np.unique(result)

    dicColors = {}
    dicColors[0] = "#000000"

    for i in range(len(uniqueValues)):
        if uniqueValues[i] != 0:
            dicColors[uniqueValues[i]] = palleteColors[i]


    colorsLeaf={}
    for i in range(len(result)):
        colorsLeaf[i] = dicColors[result[i]]

    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    linkCols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (linkCols[x] if x > len(Z) else colorsLeaf[x]
                  for x in i12)

        linkCols[i+1+len(Z)] = c1 if c1 == c2 else dicColors[0]

    fig = plt.figure(figsize=(15, 10))

    dn = dendrogram(Z=Z, color_threshold=None, leaf_font_size=5,
                     leaf_rotation=45, link_color_func=lambda x: linkCols[x])
    plt.title(title, fontsize=12)

    if saveDescription != None:
        plt.savefig(saveDescription)
        plt.close(fig)
        return

    plt.show() #\


#### ############################################  Main script #########################################################

os.makedirs("./results", exist_ok=True)

print("\nPerforming experiments in dataset\n")

dataset = Dataset("signals")

model = Autoencoder(dataset.data)
# # model.fit(dataset.data)
# # model.save("model2")
model.load("model")
prediction = model.predict(dataset.data)

modProjecao = TSNE(n_components=2, learning_rate='auto', init='random')
TSNE_2D = modProjecao.fit_transform(prediction)
clustering = DBSCAN(eps=2).fit(TSNE_2D)
TSNE_2D = pd.DataFrame(TSNE_2D)
TSNE_2D["cluster"] = clustering.labels_
TSNE_2D["label"] = list(range(prediction.shape[0]))
fig = px.scatter(TSNE_2D, x=0, y=1, color="cluster", labels="label", hover_data=["label"])
fig.write_image(f"./results/TSNE_2D.jpeg")

optics_results = OPTICS(min_samples=15).fit(prediction)
os.makedirs("./results/OPTICS", exist_ok=True)
print(f"Number of clusters in OPTICS results = {len(set(optics_results.labels_))}")
for cluster_number in set(optics_results.labels_):
    print(f"Saving OPTICS clustering results for cluster {cluster_number}")
    index_list = [i for i, item in enumerate(optics_results.labels_) if item == cluster_number]
    print(f"Number of signals in cluser {cluster_number} = {len(index_list)}")
    signal_list = dataset.original_data.iloc[index_list]
    fig = make_subplots(rows=(len(signal_list.values) // 3) + 1, cols=3)
    i = 0
    for signal in signal_list.values:
        fig.add_trace(
            go.Scatter(x=list(range(len(signal))), y=signal),
            row=((i // 3) + 1), col=((i % 3) + 1)
        )
        i += 1
    fig.update_layout(height=(((len(signal_list) // 3) + 1) * 250) + 250, title_text="Cluster " + str(cluster_number))
    fig.write_image(f"./results/OPTICS/cluster_{cluster_number}.jpeg")


hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=15)
hdbscan_results.fit(prediction)
os.makedirs("./results/HDBSCAN", exist_ok=True)
print(f"Number of clusters in HDBSCAN results = {len(set(optics_results.labels_))}")
for cluster_number in set(hdbscan_results.labels_):
    print(f"Saving HDBSCAN clustering results for cluster {cluster_number}")
    index_list = [i for i, item in enumerate(hdbscan_results.labels_) if item == cluster_number]
    print(f"Number of signals in cluser {cluster_number} = {len(index_list)}")
    signal_list = dataset.original_data.iloc[index_list]
    fig = make_subplots(rows=(len(signal_list.values) // 3) + 1, cols=3)
    i = 0
    for signal in signal_list.values:
        fig.add_trace(
            go.Scatter(x=list(range(len(signal))), y=signal),
            row=((i // 3) + 1), col=((i % 3) + 1)
        )
        i += 1
    fig.update_layout(height=(((len(signal_list) // 3) + 1) * 250) + 250, title_text="Cluster " + str(cluster_number))
    fig.write_image(f"./results/HDBSCAN/cluster_{cluster_number}.jpeg")

listOfMClSize = [25, 50, 100, 200, 300]
methodsLinkage = ["single", "average", "ward", "complete", "weighted"]

os.makedirs("./results/FOSC", exist_ok=True)
for m in listOfMClSize:
    print("--------------------------------------- MCLSIZE = %d ---------------------------------------" % m)

    for lm in methodsLinkage:

        titlePlot = lm + " and mClSize = " + str(m)
        savePath = "./results/FOSC/output/" + lm + "-" + str(m) + ".png"
        saveDendrogram = "./results/FOSC/dendrogram-" + lm + "-mClSize-" + str(m) + ".png"

        print("Using linkage method %s" % lm)
        Z = linkage(prediction, method=lm, metric="euclidean")

        print("Start FOSC")
        foscFramework = FOSC(Z, mClSize=m)
        infiniteStability = foscFramework.propagateTree()
        partition = foscFramework.findProminentClusters(1, infiniteStability)

        print("Plotting results")
        # plotPartition(mat[:,0], mat[:,1], partition, titlePlot, savePath)
        plotDendrogram(Z, partition, titlePlot, saveDendrogram)

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

sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs("./results", exist_ok=True)

dataset = Dataset("signals")

model = Autoencoder(dataset.data)
# # model.fit(dataset.data)
# # model.save("model2")
model.load("model")
prediction = model.predict(dataset.data)

modProjecao = TSNE(n_components=2, learning_rate='auto', init='random')
TSNE_2D = modProjecao.fit_transform(prediction)
TSNE_2D = pd.DataFrame(TSNE_2D)
TSNE_2D["label"] = list(range(prediction.shape[0]))

optics_results = OPTICS(min_cluster_size=50).fit(prediction)
TSNE_2D["OPTICS"] = optics_results.labels_

hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=50)
hdbscan_results.fit(prediction)
TSNE_2D["HDBSCAN"] = hdbscan_results.labels_

listOfMClSize = [25, 50, 100, 200, 300]
methodsLinkage = ["single", "average", "ward", "complete", "weighted"]

for m in listOfMClSize:
    for lm in methodsLinkage:
        titlePlot = lm + " and mClSize = " + str(m)
        Z = linkage(prediction, method=lm, metric="euclidean")
        foscFramework = FOSC(Z, mClSize=m)
        infiniteStability = foscFramework.propagateTree()
        partition = foscFramework.findProminentClusters(1, infiniteStability)
        TSNE_2D[f"FOSC_{m}_{lm}"] = partition

optics_results_fig = px.scatter(TSNE_2D, x=0, y=1, color="OPTICS", labels="label", hover_data=["label"])
optics_results_fig.write_image(f"./results/OPTICS.jpeg")
hdbscan_results_fig = px.scatter(TSNE_2D, x=0, y=1, color="HDBSCAN", labels="label", hover_data=["label"])
hdbscan_results_fig.write_image(f"./results/HDBSCAN.jpeg")
for m in listOfMClSize:
    for lm in methodsLinkage:
        hdbscan_results_fig = px.scatter(TSNE_2D, x=0, y=1, color=f"FOSC_{m}_{lm}", labels="label", hover_data=["label"])
        hdbscan_results_fig.write_image(f"./results/FOSC_{m}_{lm}.jpeg")

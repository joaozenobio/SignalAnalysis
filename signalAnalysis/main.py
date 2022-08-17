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

# TODO:
#  Report
#  for + Reachability plot OPTICS
#  for + Dendrogram HDBSCAN
#  for + Dendrogram FOSC
#  Print example "result_..." in TSNE 2D
#  Calculate metrics
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

modProjecao = TSNE(n_components=2, learning_rate='auto', init='random')
TSNE_2D = modProjecao.fit_transform(prediction)
TSNE_2D = pd.DataFrame(TSNE_2D)
TSNE_2D["label"] = list(range(prediction.shape[0]))

optics_results = OPTICS(min_samples=5, min_cluster_size=5).fit(prediction)
TSNE_2D["OPTICS"] = optics_results.labels_

hdbscan_results = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=5)
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

for m in listOfMClSize:
    for lm in methodsLinkage:
        pass

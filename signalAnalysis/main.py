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
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, silhouette_score


def computeGamma(partition, X):
    """
    Baker-Hubert Gamma Index: A measure of compactness, based on similarity between points in a cluster, compared to similarity
    with points in other clusters
    """

    if -1 in partition:
        partition = [p+1 for p in partition]

    XCopy = []
    partitionCopy=[]
    noiseSize=0

    for i in range(len(partition)):
        if partition[i] == 0:
            noiseSize+=1
        else:
            partitionCopy.append(partition[i])
            XCopy.append(X[i,:])

    if noiseSize==len(partition): return np.nan

    penalty = (len(partition)-noiseSize)/len(partition)

    XCopy = np.mat(XCopy)
    splus=0
    sminus=0
    pairDis=pdist(XCopy, "euclidean")
    numPair=len(pairDis)
    temp=np.zeros((len(partitionCopy),2))
    temp[:,0]=partitionCopy
    vecB=pdist(temp)

    #iterate through all the pairwise comparisons
    for i in range(numPair-1):
        for j in range(i+1,numPair):
            if vecB[i]>0 and vecB[j]==0:
                #heter points smaller than homo points
                if pairDis[i]<pairDis[j]:
                    splus=splus+1
                #heter points larger than homo points
                if pairDis[i]>vecB[j]:
                    sminus=sminus+1
            if vecB[i]==0 and vecB[j]>0:
                #heter points smaller than homo points
                if pairDis[j]<pairDis[i]:
                    splus=splus+1
                #heter points larger than homo points
                if pairDis[j]>vecB[i]:
                    sminus=sminus+1
    #compute the fitness
    validation = (splus-sminus)/(splus+sminus)

    return penalty*validation


def computePBandAUCCIndexes(partition, distanceMatrix):
    if -1 in partition:
        partition = [p+1 for p in partition]

    noiseSize = 0
    for value in partition:
        if value == 0:
            noiseSize += 1

    penalty = (len(partition) - noiseSize) / len(partition)

    if noiseSize == len(partition):
        return np.nan, np.nan, noiseSize, penalty

    dm = squareform(distanceMatrix)
    x = []
    yPointBiserial = []
    yAucc = []

    for i in range(len(partition)-1):
        if partition[i] == 0:
            continue

        for j in range(i, len(partition)):
            if partition[j] == 0:
                continue

            yPointBiserial.append(dm[i, j])
            yAucc.append(1/(1+dm[i, j]))

            if partition[i] == partition[j]:
                x.append(1)
            else:
                x.append(0)

    # Compute internal validity index (point biserial)
    pb,pv = pointbiserialr(x, yPointBiserial)

    # Compute area under the curve
    aucc = roc_auc_score(x, yAucc)

    return penalty*pb, penalty*aucc, noiseSize, penalty

# TODO:
#  Report
#  Calculate metrics
#  Print example "result_..." in TSNE 2D with best metrics
#  Train YOLO with new images

sys.setrecursionlimit(10000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs("./results", exist_ok=True)

dataset = Dataset("signals")

# model = Autoencoder(dataset.data)
# model.fit(dataset.data)
# model.save("model5")

model = Autoencoder()
model.load("model4")
prediction = model.predict(dataset.data)

tsne_results = TSNE(n_components=2, learning_rate='auto', init='random')
TSNE_2D = tsne_results.fit_transform(prediction)
fig, ax = plt.subplots()
fig.set_size_inches(15, 15)
plt.scatter(x=TSNE_2D[:, 0], y=TSNE_2D[:, 1])
plt.savefig(f'results/TSNE_2D', dpi=100)
plt.close(fig)

df = pd.DataFrame(columns=["Method", "Silhouette", "PBIndex", "AUCC", "GammaIndex", "Noise", "Penalty"])
dMatrix = pdist(prediction, "euclidean")

# [5, 10, 15, 20, 25, 30]
listOfMClSize = [5]
for m in listOfMClSize:
    print(f"listOfMClSize = {m}")

    print("HDBSCAN")
    hdbscan_results = hdbscan.HDBSCAN(min_cluster_size=m)
    hdbscan_results.fit(prediction)
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    hdbscan_results.condensed_tree_.plot(log_size=True)
    plt.savefig(f'results/HDBSCAN_{m}.png', dpi=100)
    plt.close(fig)
    silhouette = silhouette_score(prediction, hdbscan_results.labels_)
    pbIndex, auccIndex, noise, penalty = computePBandAUCCIndexes(hdbscan_results.labels_, dMatrix)
    # gammaIndex = computeGamma(hdbscan_results.labels_, prediction)
    df.loc[len(df)] = ["HDBSCAN_{m}.png", silhouette, pbIndex, auccIndex, 0, noise, penalty]

    print("OPTICS")
    optics_results = OPTICS(min_cluster_size=m).fit(prediction)
    reachability = optics_results.reachability_[optics_results.ordering_]
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    ax.plot(reachability)
    plt.savefig(f'results/OPTICS_{m}.png', dpi=100)
    plt.close(fig)
    silhouette = silhouette_score(prediction, optics_results.labels_)
    pbIndex, auccIndex, noise, penalty = computePBandAUCCIndexes(optics_results.labels_, dMatrix)
    # gammaIndex = computeGamma(optics_results.labels_, prediction)
    df.loc[len(df)] = ["OPTICS_{m}.png", silhouette, pbIndex, auccIndex, 0, noise, penalty]

    methodsLinkage = ["single", "average", "ward", "complete", "weighted"]
    for lm in methodsLinkage:
        print(f"FOSC - {lm}")
        titlePlot = lm + " and mClSize = " + str(m)
        Z = linkage(prediction, method=lm, metric="euclidean")
        foscFramework = FOSC(Z, mClSize=m)
        infiniteStability = foscFramework.propagateTree()
        labels_ = foscFramework.findProminentClusters(1, infiniteStability)
        fig = plt.figure(figsize=(15, 15))
        dn = dendrogram(Z=Z, color_threshold=None, leaf_font_size=5, leaf_rotation=45)
        plt.savefig(f"results/FOSC_{m}_{lm}")
        plt.close(fig)
        silhouette = silhouette_score(prediction, labels_)
        pbIndex, auccIndex, noise, penalty = computePBandAUCCIndexes(labels_, dMatrix)
        # gammaIndex = computeGamma(partition, prediction)
        df.loc[len(df)] = ["FOSC_{m}_{lm}", silhouette, pbIndex, auccIndex, 0, noise, penalty]

    print("-------------")

df.to_csv(f"results/dataframe-results.csv")

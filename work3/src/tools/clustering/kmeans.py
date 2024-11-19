from sklearn.cluster import KMeans

KMeansParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

class KMeans(KMeans):
    pass
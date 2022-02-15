import numpy as np
import cluster
import pytest

def test_sil_output_shape():
    ###Initialize data###
    clusters, labels = cluster.make_clusters(n=100, m=5, k=3, scale=1.3)
    ###Initialize clustering fxns###
    sil=cluster.Silhouette(metric="euclidean")
    kmeans=cluster.KMeans(k=3)
    ###Fit kmeans clustering to data and label###
    kmeans.fit(clusters)
    cluster_labels=kmeans.predict(clusters)
    ###calculate the silhouette score for the observations and assert we have correct shape###
    scores=sil.score(clusters, cluster_labels)
    assert scores.shape[0]==100

def test_sil_bounded_1():
    """
    Testing that silhoutte scores are correctly bounded |s|<=1
    
    """
    ###Initialize data###
    clusters, labels = cluster.make_clusters(n=100, m=5, k=3, scale=1.3)
    ###initialize clustering fxn###
    sil=cluster.Silhouette(metric="euclidean")
    kmeans=cluster.KMeans(k=3)
    ###Fit kmeans to data and get cluster labels for obs###
    kmeans.fit(clusters)
    cluster_labels=kmeans.predict(clusters)

    ###assert that silhoutte scores are bounded above and below by 1###
    scores=sil.score(clusters,cluster_labels)
    assert np.all(np.abs(scores)<=1)

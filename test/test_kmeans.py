# Write your k-means unit tests here
import numpy as np
import pytest

import cluster

def test_init():
    """
    Testing to make sure initialization function catches parameter errors
    """
    with pytest.raises(AssertionError):
        kmeans=cluster.KMeans(k=0)

    with pytest.raises(AssertionError):
        kmeans=cluster.KMeans(k=2,tol=0)

    with pytest.raises(AssertionError):
        kmeans=cluster.KMeans(k=2,max_iter=0)
    

def test_obs_greater_than_k():
    kmeans=cluster.KMeans(k=5)
    # make data
    clusters, labels = cluster.make_clusters(n=3, m=5, k=2, scale=1.3)
    with pytest.raises(AssertionError):
        kmeans.fit(clusters)

def test_n_centroids():
    clusters, labels=cluster.make_clusters(n=50,m=5, k=3)
    kmeans=cluster.KMeans(k=3)
    kmeans.fit(clusters)
    n_centroids=kmeans.get_centroids().shape[0]
    assert n_centroids == 3

def test_predict():
    clusters, labels=cluster.make_clusters(n=50,m=5, k=3)
    kmeans=cluster.KMeans(k=3)
    kmeans.fit(clusters)
    clusters_pred=kmeans.predict(clusters)
    assert len(np.unique(clusters_pred))==3

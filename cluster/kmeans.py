import numpy as np
from scipy.spatial.distance import cdist
import random

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        assert k>0, "k must be larger than 0"
        assert tol>0, "tolerance must be larger than 0"
        assert max_iter>0, "Number of iterations must be larger than 0"
        
        self.k=k
        self.metric=metric
        self.tol=tol
        self.max_iter=max_iter
    
    def _init_centroids(self,  mat: np.ndarray):
        """
        Private function to intialize centroids using the Forgy Method 
        Forgy method=take k random samples as inital centroids

        input: mat: n_obs x n_features ndarray
        output: init_centroids k x n_features
        """
        n_obs=mat.shape[0]
        
        #take random sample of obs as centroids
        random_sample=random.sample(list(np.arange(n_obs)),self.k)
        return mat[random_sample,:]
    
    def _calculate_centroid(self, mat: np.ndarray):
        """
        Private function to calculate new centroids once clusters are assigned
        
        input: mat n_obs x n_features ndarray
        output: centroids: k x n_features               
        """
        centroids=np.zeros((self.k,mat.shape[1])) #initialize centroid obj

        ##### loop thru clusters and take the mean of obs belonging to the cluster #####
        for x in range(self.k):
            cluster_points=np.where(self.cluster_id==x)
            centroids[x,:]=np.mean(mat[cluster_points,:],axis=1)
        
        return centroids
    
    def _assign_to_cluster(self,  mat: np.ndarray):
        """
        Private function to assign observations to clusters based on distance to centroid
        
        input: mat n_obs x n_features
        output: cluster_ids: n_obs x 1
        """
        ##### calculate distance of each observation to each centroid #####
        # mat \exists [n_obs,n_features] and centroids \exists [k,n_feautures]
        # --> dists \exists [n_obs,k]
        dists=cdist(mat,self.centroids, metric=self.metric)
        
        ##### assign each point to the minimum #####
        # dists \exists [n_obs,k], so take argmin over axis=1 to get index of cluster id
        return np.argmin(dists,axis=1)
    
    def fit(self, mat: np.ndarray):
        ### check parameters one more time #####
        assert mat.shape[0] > self.k, "Number of observations should be more than the number of clusters"

        ##### initalization #####
        # use Forgy method to initialize centroids
        # also need to assign large error then can min that
        self.centroids=self._init_centroids(mat)
        self.error=50
        
        for i in range(self.max_iter):
            ##### assignment step #####
            clusters=self._assign_to_cluster(mat)
            
            ##### update step #####
            self.cluster_id=clusters
            self.centroids=self._calculate_centroid(mat)
            
            ##### calculate MSE #####
            # MSE=squared average error of all points to their corresponding centroid
            mse = np.average(np.square(np.min(cdist(mat, self.centroids, metric = self.metric), axis=1)))
                
            ##### check for convergence #####
            # if mse - prev_mse < tol then we have convergence
            delta_error=np.abs(self.error-mse)
            if delta_error<self.tol:
                break
            else:
                self.error=mse
    
    def predict(self, mat: np.ndarray) -> np.ndarray:
        return self.cluster_id

    def get_error(self) -> float:

        return self.error

    def get_centroids(self) -> np.ndarray:

        return self.centroids

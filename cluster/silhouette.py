import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric=metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        
        Let i be an obs in cluster C_j
        
        Then define, 
        
        a(i) = mean distance between i and all other datapoints in the cluster C_j 
             = within cluster mean distance
        
        b(i) = smallest mean distance of i to all points of any other cluster of which i is not member
             = neighbor cluster mean distance
        
        So now the siloutte score s(i) for obs i is
        s(i) = (b(i)-a(i))/max{a(i),b(i)} if |C_j|>1
        
        and s(i) = 0 if |C_j|=1
        """
        
        k=len(np.unique(y))
        n_features=X.shape[1]
        ##### initialize siloutte score array object #####
        S=np.zeros((X.shape[0],1))
        
        for i in range(X.shape[0]):
            ##### get cluster id #####
            cluster_id=y[i]
            
            ##### check if C_i ==1 #####
            if np.sum(np.where(y==cluster_id))==1:
                continue
            
            ##### calculate within cluster mean distance ######
            # note: i don't think I need to remove i itself from the calculation b/c dist(i,i)=0
            a_i=np.average(cdist(X[np.where(y==cluster_id)[0],:],X[i,:].reshape((1,n_features)), metric=self.metric))

            ##### calculate neighbor cluster mean distance ######
            avg_dists=list()
            for j in np.unique(y):
                print
                if j!=cluster_id:
                    avg_dists.append(np.average(cdist(X[np.where(y==j)[0],:],X[i,:].reshape((1,n_features)), metric=self.metric)))
                    
            b_i=min(avg_dists)
            print(avg_dists)
            print(b_i)
            ##### calculate siloutte score #####
            s_i=(a_i-b_i)/max(a_i,b_i)
            S[i]=s_i
        
        return S
                    
            



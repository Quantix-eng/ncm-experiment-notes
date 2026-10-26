"""Archive deterministic k-means probes from their saved training geometry."""
from base import *
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
threadpool_limits(1)

def main():
    geom=np.load(D/'results/level5_geometry.npz');saved={}
    for d in ['hard','easy','medium']:
        x=geom[d+'_u5'].reshape(-1,256).astype(float)
        km=KMeans(n_clusters=16,n_init=5,max_iter=200,random_state=Q['seed']).fit(x)
        saved[d+'_centers']=km.cluster_centers_;saved[d+'_train_labels']=km.labels_
    np.savez_compressed(D/'results/kmeans_centers.npz',**saved)
    log('kmeans_centers_archived')
if __name__=='__main__':main()

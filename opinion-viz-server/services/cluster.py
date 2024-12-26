# services/cluster.py
import numpy as np
from sklearn.cluster import KMeans

def cluster_coords(coords_5d, n_clusters=3):
    """
    既にMDSで変換済みの5次元座標 coords_5d (list of list or np.array)
    に対して、KMeansクラスタリングを行い、クラスタラベルを返す。
    """
    coords_array = np.array(coords_5d)  # shape = (n_samples, 5)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords_array)  # shape = (n_samples,)
    return labels.tolist()

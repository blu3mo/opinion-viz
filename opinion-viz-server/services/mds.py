# services/mds.py
import numpy as np
from sklearn.manifold import MDS

def compute_mds_positions(distance_matrix: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    距離行列(distance_matrix)から、MDS によって n_components 次元の座標を計算する。
    dissimilarity="precomputed" を指定しているため、
    distance_matrixはサンプル間のペアワイズ距離を直接表す行列である必要がある。
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42
    )
    coords = mds.fit_transform(distance_matrix)
    return coords

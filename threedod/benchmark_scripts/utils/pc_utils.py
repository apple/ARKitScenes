import numpy as np


def down_sample(point_cloud, voxel_sz):
    """Quantize point cloud by voxel_size
    Returns kept indices

    Args:
        all_points: np.array (n, 3) float
        voxel_sz: float
    Returns:
        indices: (m, ) int
    """
    coordinates = np.round(point_cloud / voxel_sz).astype(np.int32)
    _, indices = np.unique(coordinates, axis=0, return_index=True)
    return indices
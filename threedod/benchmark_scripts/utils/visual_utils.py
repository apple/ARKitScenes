import cv2
import numpy as np

import open3d as o3d


COLOR_LIST = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (0.5, 0, 0),
    (0.6, 1, 0.3),
    (0, 0.5, 0),
    (0.5, 0, 0.5),
    (0.2, 0.3, 0.2),
    (0.8, 0.7, 0.8),
    (0.1, 0.8, 0.9),
    (0.4, 0.1, 0.7),
    (0.5, 0.5, 0),
    (0, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0, 0, 0.5),
]


def visualize_o3d(
    pc,
    boxes,
    pc_color=None,
    width=384,
    height=288,
):
    """
    Visualize result with open3d
    Args:
        pc: np.array of shape (n, 3)
            point cloud
        boxes: a list of m boxes, each item as a tuple:
            (cls, np.array (8, 3), conf) if predicted
            (cls, np.array (8, 3))
            or just np.array (n, 8, 3)
        pc_color: np.array (n, 3) or None
        box_color: np.array (m, 3) or None
        visualize: bool (directly visualize or return an image)
        width: int
            used only when visualize=False
        height: int
            used only when visualize=False
    Returns:
    """
    assert pc.shape[1] == 3
    ratio = max(1, pc.shape[0] // 4000)
    pc_sample = pc[::ratio, :]

    n = pc_sample.shape[0]
    m = len(boxes)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_sample)
    if pc_color is None:
        pc_color = np.zeros((n, 3))
    pcd.colors = o3d.utility.Vector3dVector(pc_color)

    linesets = []
    for i, item in enumerate(boxes):
        if isinstance(item, tuple):
            cls_ = item[0]
            assert isinstance(cls_, int)
            corners = item[1]
        else:
            cls_ = None
            corners = item
        assert corners.shape[0] == 8
        assert corners.shape[1] == 3

        if isinstance(cls_, int) and cls_ < len(COLOR_LIST):
            tmp_color = COLOR_LIST[cls_]
        else:
            tmp_color = (0, 0, 0)
        linesets.append(get_lines(corners, color=tmp_color))

    o3d.visualization.draw_geometries([pcd] + linesets)
    return None


def get_lines(box, color=np.array([1.0, 0.0, 0.0])):
    """
    Args:
        box: np.array (8, 3)
            8 corners
        color: line color
    Returns:
        o3d.Linset()
    """
    points = box
    lines = [
        [0, 1],
        [0, 3],
        [1, 2],
        [2, 3],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# TODO: Explain 8 corners logic at the top and use it consistently
# Add comments of explanation

import numpy as np
import scipy.spatial

from .rotation import rotate_points_along_z


def get_size(box):
    """
    Args:
        box: 8x3
    Returns:
        size: [dx, dy, dz]
    """
    distance = scipy.spatial.distance.cdist(box[0:1, :], box[1:5, :])
    l = distance[0, 2]
    w = distance[0, 0]
    h = distance[0, 3]
    return [l, w, h]


def get_heading_angle(box):
    """
    Args:
        box: (8, 3)
    Returns:
        heading_angle: float
    """
    a = box[0, 0] - box[1, 0]
    b = box[0, 1] - box[1, 1]

    heading_angle = np.arctan2(a, b)
    return heading_angle


def compute_box_3d(size, center, rotmat):
    """Compute corners of a single box from rotation matrix
    Args:
        size: list of float [dx, dy, dz]
        center: np.array [x, y, z]
        rotmat: np.array (3, 3)
    Returns:
        corners: (8, 3)
    """
    l, h, w = [i / 2 for i in size]
    center = np.reshape(center, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(
        np.transpose(rotmat), np.vstack([x_corners, y_corners, z_corners])
    )
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def corners_to_boxes(corners3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        corners: (N, 8, 3), vertex order shown in figure above

    Returns:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading]
            with (x, y, z) is the box center
            (dx, dy, dz) as the box size
            and heading as the clockwise rotation angle
    """

    boxes3d = np.zeros((corners3d.shape[0], 7))
    for i in range(corners3d.shape[0]):
        boxes3d[i, :3] = np.mean(corners3d[i, :, :], axis=0)
        boxes3d[i, 3:6] = get_size(corners3d[i, :, :])
        boxes3d[i, 6] = get_heading_angle(corners3d[i, :, :])

    return boxes3d


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center

    Returns:
        corners: (N, 8, 3)
    """
    template = np.array([[1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1]]
    ) / 2.

    # corners3d: of shape (N, 3, 8)
    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]

    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(
        -1, 8, 3
    )
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def points_in_boxes(points, boxes):
    """
    Args:
        pc: np.array (n, 3+d)
        boxes: np.array (m, 8, 3)
    Returns:
        mask: np.array (n, m) of type bool
    """
    if len(boxes) == 0:
        return np.zeros([points.shape[0], 1], dtype=np.bool)
    points = points[:, :3]  # get xyz
    # u = p6 - p5
    u = boxes[:, 6, :] - boxes[:, 5, :]  # (m, 3)
    # v = p6 - p7
    v = boxes[:, 6, :] - boxes[:, 7, :]  # (m, 3)
    # w = p6 - p2
    w = boxes[:, 6, :] - boxes[:, 2, :]  # (m, 3)

    # ux, vx, wx
    ux = np.matmul(points, u.T)  # (n, m)
    vx = np.matmul(points, v.T)
    wx = np.matmul(points, w.T)

    # up6, up5, vp6, vp7, wp6, wp2
    up6 = np.sum(u * boxes[:, 6, :], axis=1)
    up5 = np.sum(u * boxes[:, 5, :], axis=1)
    vp6 = np.sum(v * boxes[:, 6, :], axis=1)
    vp7 = np.sum(v * boxes[:, 7, :], axis=1)
    wp6 = np.sum(w * boxes[:, 6, :], axis=1)
    wp2 = np.sum(w * boxes[:, 2, :], axis=1)

    mask_u = np.logical_and(ux <= up6, ux >= up5)  # (1024, n)
    mask_v = np.logical_and(vx <= vp6, vx >= vp7)
    mask_w = np.logical_and(wx <= wp6, wx >= wp2)

    mask = mask_u & mask_v & mask_w  # (10240, n)

    return mask


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = scipy.spatial.ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3,-1,-1)]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[:,2].max(), corners2[:,2].max())
    ymin = max(corners1[:,2].min(), corners2[:,2].min())
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou
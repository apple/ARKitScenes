import numpy as np


def decide_pose(pose):
    """
    Args:
        pose: np.array (4, 4)
    Returns:
        index: int (0, 1, 2, 3)
        for upright, left, upside-down and right
    """
    # pose style
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0, 1.0, 0.0],  # upside-down
            [1.0, 0.0, 0.0],
        ]  # right
    )
    corr = np.matmul(z_orien, z_vec)
    corr_max = np.argmax(corr)
    return corr_max


def rotate_pose(im, rot_index):
    """
    Args:
        im: (m, n)
    """
    h, w, d = im.shape
    if d == 3:
        if rot_index == 0:
            new_im = im
        elif rot_index == 1:
            new_im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot_index == 2:
            new_im = cv2.rotate(im, cv2.ROTATE_180)
        elif rot_index == 3:
            new_im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_im


# rotate image
orientation_index = decide_pose(pose)
im = rotate_pose(im, corr_max_index)
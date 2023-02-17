import cv2
import numpy as np

from utils.tenFpsDataLoader import TrajStringToMatrix


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


# rectify image: case 1
# scene-id: 41048143 downloaded by
#   python download_data.py 3dod --split Training --video_id 41048143
# the 1st line (frame#1) from lowres_wide.traj
line = "1627.68140546 -1.5545600446434271 1.532901960141653 0.883467254271724 0.0450016 0.0506107 -0.000194237"
traj_timestamp = line.split(" ")[0]
pose = TrajStringToMatrix(line)[1]  # parse the pose matrix
orientation_index = decide_pose(pose)  # index 2, means image should be rotated 180-deg
print(orientation_index)
# im = rotate_pose(im, orientation_index)  # choose frame#1, image will be rectified

# rectify image: case 2
# scene-id: 41048169
#   python download_data.py 3dod --split Training --video_id 41048169
# the 1st line (frame#1) from lowres_wide.traj
line = "803.47236621 1.6851560664954446 -1.7402208764128138 -0.8469396625258023 0.0404551 0.0562208 -0.00155703"
traj_timestamp = line.split(" ")[0]
pose = TrajStringToMatrix(line)[1]  # parse the pose matrix
orientation_index = decide_pose(pose)  # index 1, means image should be rotated 90-deg clockwise
print(orientation_index)
# im = rotate_pose(im, orientation_index)  # choose frame#1, image will be rectified

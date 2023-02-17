import copy
import cv2
import glob
import json
import numpy as np
import os

from .box_utils import compute_box_3d, boxes_to_corners_3d, get_size
from .rotation import convert_angle_axis_to_matrix3
from .taxonomy import class_names, ARKitDatasetConfig


def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def generate_point(
    rgb_image,
    depth_image,
    intrinsic,
    subsample=1,
    world_coordinate=True,
    pose=None,
):
    """Generate 3D point coordinates and related rgb feature
    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth
        intrinsic: (3, 3)
        subsample: int
            resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate
    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack(
            (new_points_3d, np.ones((1, new_points_3d.shape[1])))
        )
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, rgb_feat


def extract_gt(gt_fn):
    """extract original label data

    Args:
        gt_fn: str (file name of "annotation.json")
            after loading, we got a dict with keys
                'data', 'stats', 'comment', 'confirm', 'skipped'
            ['data']: a list of dict for bboxes, each dict has keys:
                'uid', 'label', 'modelId', 'children', 'objectId',
                'segments', 'hierarchy', 'isInGroup', 'labelType', 'attributes'
                'label': str
                'segments': dict for boxes
                    'centroid': list of float (x, y, z)?
                    'axesLengths': list of float (x, y, z)?
                    'normalizedAxes': list of float len()=9
                'uid'
            'comments':
            'stats': ...
    Returns:
        skipped: bool
            skipped or not
        boxes_corners: (n, 8, 3) box corners
            **world-coordinate**
        centers: (n, 3)
            **world-coordinate**
        sizes: (n, 3) full-sizes (no halving!)
        labels: list of str
        uids: list of str
    """
    gt = json.load(open(gt_fn, "r"))
    skipped = gt['skipped']
    if len(gt) == 0:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        return skipped, boxes_corners, centers, sizes, labels, uids

    boxes_corners = []
    centers = []
    sizes = []
    labels = []
    uids = []
    for data in gt['data']:
        l = data["label"]
        for delimiter in [" ", "-", "/"]:
            l = l.replace(delimiter, "_")
        if l not in class_names:
            print("unknown category: %s" % l)
            continue

        rotmat = np.array(data["segments"]["obbAligned"]["normalizedAxes"]).reshape(
            3, 3
        )
        center = np.array(data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        size = np.array(data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(size.reshape(3).tolist(), center, rotmat)

        '''
            Box corner order that we return is of the format below:
                6 -------- 7
               /|         /|
              5 -------- 4 .
              | |        | |
              . 2 -------- 3
              |/         |/
              1 -------- 0 
        '''

        boxes_corners.append(box3d.reshape(1, 8, 3))
        size = np.array(get_size(box3d)).reshape(1, 3)
        center = np.mean(box3d, axis=0).reshape(1, 3)

        # boxes_corners.append(box3d.reshape(1, 8, 3))
        centers.append(center)
        sizes.append(size)
        # labels.append(l)
        labels.append(data["label"])
        uids.append(data["uid"])
    centers = np.concatenate(centers, axis=0)
    sizes = np.concatenate(sizes, axis=0)
    boxes_corners = np.concatenate(boxes_corners, axis=0)
    return skipped, boxes_corners, centers, sizes, labels, uids


class TenFpsDataLoader(object):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        root_path=None,
        gt_path=None,
        logger=None,
        frame_rate=1,
        with_color_image=True,
        subsample=2,
        world_coordinate=True,
    ):
        """
        Args:
            dataset_cfg: EasyDict() with key
                POINT_CLOUD_RANGE
                POINT_FEATURE_ENCODING
                DATA_PROCESSOR
            class_names: list of str
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
            gt_path: xxx.json
                just to get correct floor height
            an2d_root: path to scene_id.json
                or None
            logger:
            frame_rate: int
            subsample: int
            world_coordinate: bool
        """
        self.root_path = root_path

        # pipeline does box residual coding here
        self.num_class = len(class_names)

        self.dc = ARKitDatasetConfig()

        depth_folder = os.path.join(self.root_path, "lowres_depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
            self.frame_ids = [os.path.basename(x) for x in depth_images]
            self.frame_ids = [x.split(".png")[0].split("_")[1] for x in self.frame_ids]
            self.video_id = depth_folder.split('/')[-3]
            self.frame_ids = [x for x in self.frame_ids]
            self.frame_ids.sort()
            self.intrinsics = {}

        traj_file = os.path.join(self.root_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            self.traj = f.readlines()
        # convert traj to json dict
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

        if os.path.exists(traj_file):
            # self.poses = json.load(open(traj_file))
            self.poses = poses_from_traj
        else:
            self.poses = {}

        # get intrinsics
        for frame_id in self.frame_ids:
            intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics", f"{self.video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            self.intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)

        # # intrinsic_fn = os.path.join(self.root_path, "camera.txt")
        # intrinsic_fn = os.path.join(self.root_path, "color.pincam")
        # if os.path.exists(intrinsic_fn):
        #     self.intrinsics = st2_camera_intrinsics(intrinsic_fn)
        # else:
        #     self.intrinsics = None

        self.frame_rate = frame_rate
        self.subsample = subsample
        self.with_color_image = with_color_image
        self.world_coordinate = world_coordinate

        if gt_path is not None and os.path.exists(gt_path):
            skipped, gt_corners, gt_centers, gt_sizes, _, _ = extract_gt(gt_path)
            self.gt_corners = gt_corners
            self.gt_centers = gt_centers
            self.gt_sizes = gt_sizes
        else:
            self.gt_corners = None
            self.gt_centers = None
            self.gt_sizes = None

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        fname = "{}_{}.png".format(self.video_id, frame_id)
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path)

        image_path = os.path.join(self.root_path, "lowres_wide", fname)

        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        frame["depth"] = cv2.imread(depth_image_path, -1)
        frame["image"] = cv2.imread(image_path)
        frame["image_path"] = image_path
        depth_height, depth_width = frame["depth"].shape
        im_height, im_width, im_channels = frame["image"].shape

        frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id])
        if str(frame_id) in self.poses.keys():
            frame_pose = np.array(self.poses[str(frame_id)])
        else:
            for my_key in list(self.poses.keys()):
                if abs(float(frame_id) - float(my_key)) < 0.005:
                    frame_pose = np.array(self.poses[str(my_key)])
        frame["pose"] = copy.deepcopy(frame_pose)

        im_height_scale = np.float(depth_height) / im_height
        im_width_scale = np.float(depth_width) / im_width

        if depth_height != im_height:
            frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.imread(image_path)

        (m, n, _) = frame["image"].shape
        depth_image = frame["depth"] / 1000.0
        rgb_image = frame["image"] / 255.0

        pcd, rgb_feat = generate_point(
            rgb_image,
            depth_image,
            frame["intrinsics"],
            self.subsample,
            self.world_coordinate,
            frame_pose,
        )

        frame["pcd"] = pcd
        frame["color"] = rgb_feat
        return frame
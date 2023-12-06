import os

import mmcv
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


DATA_PATH = 'data/nuScenes'
RADAR_SPLIT = 'radar_bev_filter'
OUT_PATH = 'radar_pv_filter'
info_paths = ['data/nuScenes/nuscenes_infos_train.pkl', 'data/nuScenes/nuscenes_infos_val.pkl']

# DATA_PATH = 'data/nuScenes/v1.0-test'
# RADAR_SPLIT = 'radar_bev_filter_test'
# OUT_PATH = 'radar_pv_filter_test'
# info_paths = ['data/nuScenes/nuscenes_infos_test.pkl']

MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.

IMG_SHAPE = (900, 1600)

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    features,
    img_shape,
    cam_calibrated_sensor,
    cam_ego_pose,
):
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    features = np.concatenate((depths[:, None], features), axis=1)

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > MIN_DISTANCE)
    mask = np.logical_and(mask, depths < MAX_DISTANCE)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img_shape[0] - 1)
    points = points[:, mask]
    features = features[mask]

    return points, features


def worker(info):
    radar_file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    points = np.fromfile(os.path.join(DATA_PATH, RADAR_SPLIT, radar_file_name),
                         dtype=np.float32,
                         count=-1).reshape(-1, 7)

    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.
    pc = LidarPointCloud(points[:, :4].T)  # use 4 dims for code compatibility
    features = points[:, 3:]

    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        pts_img, features_img = map_pointcloud_to_image(
            pc.points.copy(), features.copy(), IMG_SHAPE, cam_calibrated_sensor, cam_ego_pose)

        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, features_img],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(DATA_PATH, OUT_PATH,
                                        f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")


if __name__ == '__main__':
    mmcv.mkdir_or_exist(os.path.join(DATA_PATH, OUT_PATH))
    for info_path in info_paths:
        infos = mmcv.load(info_path)
        for info in tqdm(infos):
            worker(info)

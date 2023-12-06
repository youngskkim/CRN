import os
from functools import reduce

import mmcv
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points


SPLIT = 'v1.0-trainval'
DATA_PATH = 'data/nuScenes'
OUT_PATH = 'radar_bev_filter'
info_paths = ['data/nuScenes/nuscenes_infos_train.pkl', 'data/nuScenes/nuscenes_infos_val.pkl']

# SPLIT = 'v1.0-test'
# DATA_PATH = 'data/nuScenes/v1.0-test'
# OUT_PATH = 'radar_bev_filter_test'
# info_paths = ['data/nuScenes/nuscenes_infos_test.pkl']

RADAR_CHAN = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
              'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

N_SWEEPS = 8
MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.

DISABLE_FILTER = False
DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    from nuscenes.utils.data_classes import LidarPointCloud

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L315
# FIELDS: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state 
#         x_rms y_rms invalid_state pdh0 vx_rms vy_rms
SAVE_FIELDS = [0, 1, 2, 5, 8, 9, -1]  # x, y, z, rcs, vx_comp, vy_comp, (dummy field for sweep info)

nusc = NuScenes(
    version=SPLIT, dataroot=DATA_PATH, verbose=True)

if DISABLE_FILTER:
    # use all point
    invalid_states = list(range(18))
    dynprop_states = list(range(8))
    ambig_states = list(range(5))
else:
    # use filtered point by invalid states and ambiguity status
    invalid_states = [0, 4, 8, 9, 10, 11, 12, 15, 16, 17]
    dynprop_states = list(range(8))
    ambig_states = [3]


def worker(info):
    # Init.
    points = np.zeros((18, 0))
    all_pc = RadarPointCloud(points)

    sample = nusc.get('sample', info['lidar_infos']['LIDAR_TOP']['sample_token'])
    lidar_data = info['lidar_infos']['LIDAR_TOP']
    ref_pose_rec = nusc.get('ego_pose', lidar_data['ego_pose']['token'])
    ref_cs_rec = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor']['token'])
    ref_time = 1e-6 * lidar_data['timestamp']
    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                    Quaternion(ref_cs_rec['rotation']), inverse=True)
    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ref_pose_rec['translation'],
                                       Quaternion(ref_pose_rec['rotation']), inverse=True)

    if DEBUG:
        lidar = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))

    for chan in RADAR_CHAN:
        # Aggregate current and previous sweeps.
        sample_data_token = sample['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for sweep in range(N_SWEEPS):
            # Load up the pointcloud and remove points close to the sensor.
            file_name = os.path.join(nusc.dataroot, current_sd_rec['filename'])
            current_pc = RadarPointCloud.from_file(file_name, invalid_states, dynprop_states, ambig_states)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.

            # Rotate velocity to reference frame
            velocities = current_pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(current_pc.points.shape[1])))
            velocities = np.dot(Quaternion(current_cs_rec['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_rec['rotation']).rotation_matrix.T, velocities)
            current_pc.points[8:10, :] = velocities[0:2, :]

            # Compensate points on moving object by velocity of point
            current_pc.points[0, :] += current_pc.points[8, :] * time_lag
            current_pc.points[1, :] += current_pc.points[9, :] * time_lag

            # Save sweep index to unused field
            current_pc.points[-1, :] = sweep

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    points = all_pc.points[SAVE_FIELDS, :].T.astype(np.float32)

    dist = np.linalg.norm(points[:, 0:2], axis=1)
    mask = np.ones(dist.shape[0], dtype=bool)
    mask = np.logical_and(mask, dist > MIN_DISTANCE)
    mask = np.logical_and(mask, dist < MAX_DISTANCE)
    points = points[mask, :]

    file_name = os.path.split(info['lidar_infos']['LIDAR_TOP']['filename'])[-1]
    points.tofile(os.path.join(DATA_PATH, OUT_PATH, file_name))

    if DEBUG:
        fig, ax = plt.subplots(figsize=(12, 12))
        points = all_pc.points[:3, :]

        velocities = all_pc.points[8:11, :]
        velocities[2, :] = np.zeros(all_pc.points.shape[1])
        viewpoint = np.eye(4)
        points_vel = view_points(all_pc.points[:3, :] + velocities, viewpoint, normalize=False)
        deltas_vel = points_vel - points
        deltas_vel = 6 * deltas_vel  # Arbitrary scaling
        max_delta = 20
        deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
        for i in range(points.shape[1]):
          ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i])

        ax.scatter(all_pc.points[0, :], all_pc.points[1, :], c='red', s=2)
        ax.plot(lidar.points[0, :], lidar.points[1, :], ',')
        plt.xlim([-60, 60])
        plt.ylim([-60, 60])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


if __name__ == '__main__':
    mmcv.mkdir_or_exist(os.path.join(DATA_PATH, OUT_PATH))
    for info_path in info_paths:
        infos = mmcv.load(info_path)
        for info in tqdm(infos):
            worker(info)


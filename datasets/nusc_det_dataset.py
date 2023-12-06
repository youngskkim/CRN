import os

import torch
import mmcv
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box
from torch.utils.data import Dataset

__all__ = ['NuscDatasetRadarDet']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_det_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


class NuscDatasetRadarDet(Dataset):
    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 rda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 load_interval=1,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 img_backbone_conf=dict(
                     x_bound=[-51.2, 51.2, 0.8],
                     y_bound=[-51.2, 51.2, 0.8],
                     z_bound=[-5, 3, 8],
                     d_bound=[2.0, 58.0, 0.5]
                 ),
                 drop_aug_conf=None,
                 return_image=True,
                 return_depth=False,
                 return_radar_pv=False,
                 depth_path='depth_gt',
                 radar_pv_path='radar_pv_filter',
                 remove_z_axis=False,
                 use_cbgs=False,
                 gt_for_radar_only=False,
                 sweep_idxes=list(),
                 key_idxes=list()):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(mmcv.load(info_path))
        else:
            self.infos = mmcv.load(info_paths)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.rda_aug_conf = rda_aug_conf
        self.drop_aug_conf = drop_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.img_backbone_conf = img_backbone_conf

        self.return_image = return_image
        self.return_depth = return_depth
        self.return_radar_pv = return_radar_pv

        self.remove_z_axis = remove_z_axis
        self.gt_for_radar_only = gt_for_radar_only

        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes
        if load_interval > 1:
            self.infos = self.infos[::load_interval]
        self.depth_path = depth_path
        self.radar_pv_path = radar_pv_path

        self.max_radar_points_pv = 1536
        self.max_distance_pv = self.img_backbone_conf['d_bound'][1]

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            if np.random.uniform() < self.bda_aug_conf['rot_ratio']:
                rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            else:
                rotate_bda = 0
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def sample_radar_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            radar_idx = np.random.choice(self.rda_aug_conf['N_sweeps'],
                                         self.rda_aug_conf['N_use'],
                                         replace=False)
        else:
            radar_idx = np.arange(self.rda_aug_conf['N_sweeps'])
        return radar_idx

    def transform_radar_pv(self, points, resize, resize_dims, crop, flip, rotate, radar_idx):
        points = points[points[:, 2] < self.max_distance_pv, :]

        H, W = resize_dims
        points[:, :2] = points[:, :2] * resize
        points[:, 0] -= crop[0]
        points[:, 1] -= crop[1]
        if flip:
            points[:, 0] = resize_dims[1] - points[:, 0]

        points[:, 0] -= W / 2.0
        points[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        points[:, :2] = np.matmul(rot_matrix, points[:, :2].T).T

        points[:, 0] += W / 2.0
        points[:, 1] += H / 2.0

        depth_coords = points[:, :2].astype(np.int16)

        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                      & (depth_coords[:, 0] < resize_dims[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))

        points = torch.Tensor(points[valid_mask])

        if self.remove_z_axis:
            points[:, 1] = 1.  # dummy height value

        points_save = []
        for i in radar_idx:
            points_save.append(points[points[:, 6] == i])
        points = torch.cat(points_save, dim=0)

        # mean, std of rcs and speed are from train set
        points[:, 3] = (points[:, 3] - 4.783) / 7.576
        points[:, 4] = (torch.norm(points[:, 4:6], dim=1) - 0.677) / 1.976

        if self.is_train:
            drop_idx = np.random.uniform(size=points.shape[0])  # randomly drop points
            points = points[drop_idx > self.rda_aug_conf['drop_ratio']]

        num_points, num_feat = points.shape
        if num_points > self.max_radar_points_pv:
            choices = np.random.choice(num_points, self.max_radar_points_pv, replace=False)
            points = points[choices]
        else:
            num_append = self.max_radar_points_pv - num_points
            points = torch.cat([points, -999*torch.ones(num_append, num_feat)], dim=0)

        if num_points == 0:
            points[0, :] = points.new_tensor([0.1, 0.1, self.max_distance_pv-1, 0, 0, 0, 0])

        points[..., [0, 1, 2]] = points[..., [0, 2, 1]]  # convert [w, h, d] to [w, d, h]

        return points[..., :5]

    def depth_transform(self, cam_depth, resize, resize_dims, crop, flip, rotate):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (tuple): Final dimension.
            crop (tuple): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """
        valid_depth = cam_depth[:, 2] < self.img_backbone_conf['d_bound'][1]
        cam_depth = cam_depth[valid_depth, :]

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(resize_dims)
        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                      & (depth_coords[:, 0] < resize_dims[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                  depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

        return torch.Tensor(depth_map)

    def get_image(self, cam_infos, cams):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_gt_depths = list()
        sweep_radar_points = list()
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            gt_depths = list()
            radar_points = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation()
            radar_idx = self.sample_radar_augmentation()

            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))

                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])

                file_name = os.path.split(cam_info[cam]['filename'])[-1]
                if self.return_depth:
                    point_depth = np.fromfile(os.path.join(
                        self.data_root, self.depth_path, f'{file_name}.bin'),
                                              dtype=np.float32,
                                              count=-1)
                    point_depth = point_depth.reshape(-1, 3)
                    point_depth_augmented = self.depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    gt_depths.append(point_depth_augmented)

                if self.return_radar_pv:
                    radar_point = np.fromfile(os.path.join(
                        self.data_root, self.radar_pv_path, f'{file_name}.bin'),
                        dtype=np.float32,
                        count=-1).reshape(-1, 7)
                    radar_point_augmented = self.transform_radar_pv(
                        radar_point, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida, radar_idx)
                    radar_points.append(radar_point_augmented)

                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_gt_depths.append(torch.stack(gt_depths))
            if self.return_radar_pv:
                sweep_radar_points.append(torch.stack(radar_points))

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_gt_depths).permute(1, 0, 2, 3),)
        else:
            ret_list.append(None)
        if self.return_radar_pv:
            ret_list.append(torch.stack(sweep_radar_points).permute(1, 0, 2, 3),)
        else:
            ret_list.append(None)
        return ret_list

    def get_image_meta(self, cam_infos, cams):
        key_info = cam_infos[0]

        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )
        return img_metas

    def get_image_sensor2ego_mats(self, cam_infos, cams):
        sweep_sensor2ego_mats = list()
        for cam in cams:
            sensor2ego_mats = list()
            key_info = cam_infos[0]
            for sweep_idx, cam_info in enumerate(cam_infos):
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
        return torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3)

    def get_gt(self, info, cams, return_corners=False):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        if return_corners:  # for debugging and visualization
            gt_corners = list()
        else:
            gt_corners = None
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if self.gt_for_radar_only:
                if ann_info['num_radar_pts'] == 0:
                    continue
            if map_name_from_general_to_detection[ann_info['category_name']] not in self.classes:
                continue
            if ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] == 0:
                continue

            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
            if return_corners:  # for debugging and visualization
                gt_corners.append(box.corners())

        return torch.Tensor(gt_boxes), torch.tensor(gt_labels), gt_corners

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        pts_infos = list()
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            while self.infos[cur_idx]['scene_token'] != self.infos[idx]['scene_token']:
                cur_idx += 1
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            pts_infos.append([info['lidar_infos']] + info['lidar_sweeps'])
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx), -1,
                                   -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            break

        if self.return_image or self.return_depth or self.return_radar_pv:
            image_data_list = self.get_image(cam_infos, cams)
            (
                sweep_imgs,
                sweep_sensor2ego_mats,
                sweep_intrins,
                sweep_ida_mats,
                sweep_sensor2sensor_mats,
                sweep_timestamps,
            ) = image_data_list[:6]
        else:
            (
                sweep_imgs,
                sweep_intrins,
                sweep_ida_mats,
                sweep_sensor2sensor_mats,
                sweep_timestamps,
            ) = None, None, None, None, None
            sweep_sensor2ego_mats = self.get_image_sensor2ego_mats(cam_infos, cams)

        img_metas = self.get_image_meta(cam_infos, cams)
        img_metas['token'] = self.infos[idx]['sample_token']
        gt_boxes_3d, gt_labels_3d, gt_corners = self.get_gt(self.infos[idx], cams, return_corners=False)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        gt_boxes_3d, bda_rot = bev_det_transform(gt_boxes_3d, rotate_bda, scale_bda, flip_dx, flip_dy)

        bda_mat = torch.zeros(4, 4, dtype=torch.float32)
        bda_mat[:3, :3] = bda_rot
        bda_mat[3, 3] = 1

        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes_3d,
            gt_labels_3d,
        ]

        if self.return_depth:
            ret_list.append(image_data_list[6])
        else:
            ret_list.append(None)
        if self.return_radar_pv:
            ret_list.append(image_data_list[7])
        else:
            ret_list.append(None)

        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)


def collate_fn(data,
               is_return_image=True,
               is_return_depth=False,
               is_return_radar_pv=False):
    assert (is_return_image or is_return_depth or is_return_radar_pv) is True
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    gt_boxes_3d_batch = list()
    gt_labels_3d_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    radar_pv_batch = list()

    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
        if is_return_radar_pv:
            radar_pv = iter_data[11]
            radar_pv_batch.append(radar_pv)

        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        img_metas_batch.append(img_metas)
        gt_boxes_3d_batch.append(gt_boxes)
        gt_labels_3d_batch.append(gt_labels)

    if is_return_image:
        mats_dict = dict()
        mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
        mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
        mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
        mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
        mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
        ret_list = [
            torch.stack(imgs_batch),
            mats_dict,
            img_metas_batch,
            gt_boxes_3d_batch,
            gt_labels_3d_batch,
            None,  # reserve for segmentation
        ]
    else:
        ret_list = [
            None,
            None,
            img_metas_batch,
            gt_boxes_3d_batch,
            gt_labels_3d_batch,
            None,
        ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    else:
        ret_list.append(None)
    if is_return_radar_pv:
        ret_list.append(torch.stack(radar_pv_batch))
    else:
        ret_list.append(None)

    return ret_list

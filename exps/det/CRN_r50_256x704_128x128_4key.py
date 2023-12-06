"""
mAP: 0.4725
mATE: 0.5190
mASE: 0.2819
mAOE: 0.4936
mAVE: 0.2732
mAAE: 0.1779
NDS: 0.5617
Eval time: 182.7s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.719	0.307	0.169	0.129	0.296	0.192
truck	0.427	0.498	0.219	0.140	0.231	0.190
bus	0.533	0.565	0.202	0.093	0.389	0.202
trailer	0.239	0.834	0.255	0.671	0.189	0.075
construction_vehicle	0.158	0.857	0.500	1.289	0.124	0.354
pedestrian	0.462	0.561	0.293	0.668	0.358	0.178
motorcycle	0.541	0.396	0.258	0.599	0.426	0.215
bicycle	0.459	0.400	0.258	0.707	0.173	0.016
traffic_cone	0.577	0.397	0.390	nan	nan	nan
barrier	0.610	0.375	0.276	0.146	nan	nan

img: 18.66
  img_backbone: 11.22
  img_dep: 1.51
  img_transform: 5.15
  img_pool: 0.54
pts: 10.62
  pts_voxelize: 1.92
  pts_backbone: 6.79
  pts_head: 1.31
fusion: 6.80
  fusion_pre: 0.81
  fusion_layer: 5.34
  fusion_post: 0.06
head: 7.90
  head_backbone: 2.46
  head_head: 5.44
total: 43.97

FPS: 22.74

   | Name                                    | Type                      | Params
---------------------------------------------------------------------------------------
0  | model                                   | CameraRadarNetDet         | 61.4 M
1  | model.backbone_img                      | RVTLSSFPN                 | 31.6 M
2  | model.backbone_img.img_backbone         | ResNet                    | 23.5 M
3  | model.backbone_img.img_neck             | SECONDFPN                 | 2.0 M
4  | model.backbone_img.depth_net            | DepthNet                  | 5.4 M
5  | model.backbone_img.view_aggregation_net | ViewAggregation           | 807 K
6  | model.backbone_pts                      | PtsBackbone               | 5.7 M
7  | model.backbone_pts.pts_voxel_layer      | Voxelization              | 0
8  | model.backbone_pts.pts_voxel_encoder    | PillarFeatureNet          | 2.3 K
9  | model.backbone_pts.pts_middle_encoder   | PointPillarsScatter       | 0
10 | model.backbone_pts.pts_backbone         | SECOND                    | 4.2 M
11 | model.backbone_pts.pts_neck             | SECONDFPN                 | 180 K
12 | model.backbone_pts.pred_context         | Sequential                | 679 K
13 | model.backbone_pts.pred_occupancy       | Sequential                | 664 K
14 | model.fuser                             | MFAFuser                  | 1.2 M
15 | model.fuser.norm_img                    | LayerNorm                 | 160
16 | model.fuser.norm_pts                    | LayerNorm                 | 160
17 | model.fuser.input_proj                  | Linear                    | 20.6 K
18 | model.fuser.positional_encoding         | LearnedPositionalEncoding | 16.4 K
19 | model.fuser.ffn_layers                  | ModuleList                | 395 K
20 | model.fuser.norm_layers1                | ModuleList                | 1.5 K
21 | model.fuser.norm_layers2                | ModuleList                | 1.5 K
22 | model.fuser.attn_layers                 | ModuleList                | 198 K
23 | model.fuser.reduce_conv                 | Sequential                | 590 K
24 | model.head                              | BEVDepthHead              | 22.8 M
25 | model.head.loss_cls                     | GaussianFocalLoss         | 0
26 | model.head.loss_bbox                    | L1Loss                    | 0
27 | model.head.shared_conv                  | ConvModule                | 147 K
28 | model.head.task_heads                   | ModuleList                | 1.4 M
29 | model.head.trunk                        | ResNet                    | 18.3 M
30 | model.head.neck                         | SECONDFPN                 | 3.0 M
---------------------------------------------------------------------------------------
"""
import torch
from utils.torch_dist import synchronize

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel

from models.camera_radar_net_det import CameraRadarNetDet


class CRNLightningModel(BEVDepthLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.return_image = True
        self.return_depth = True
        self.return_radar_pv = True
        ################################################
        self.optimizer_config = dict(
            type='AdamW',
            lr=2e-4,
            weight_decay=1e-4)
        ################################################
        self.ida_aug_conf = {
            'resize_lim': (0.386, 0.55),
            'final_dim': (256, 704),
            'rot_lim': (0., 0.),
            'H': 900,
            'W': 1600,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.0),
            'cams': [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            'Ncams': 6,
        }
        self.bda_aug_conf = {
            'rot_ratio': 1.0,
            'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.9, 1.1),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        }
        ################################################
        self.backbone_img_conf = {
            'x_bound': [-51.2, 51.2, 0.8],
            'y_bound': [-51.2, 51.2, 0.8],
            'z_bound': [-5, 3, 8],
            'd_bound': [2.0, 58.0, 0.8],
            'final_dim': (256, 704),
            'downsample_factor': 16,
            'img_backbone_conf': dict(
                type='ResNet',
                depth=50,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            ),
            'img_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[128, 128, 128, 128],
            ),
            'depth_net_conf':
                dict(in_channels=512, mid_channels=256),
            'radar_view_transform': True,
            'camera_aware': False,
            'output_channels': 80,
        }
        ################################################
        self.backbone_pts_conf = {
            'pts_voxel_layer': dict(
                max_num_points=8,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                max_voxels=(768, 1024)
            ),
            'pts_voxel_encoder': dict(
                type='PillarFeatureNet',
                in_channels=5,
                feat_channels=[32, 64],
                with_distance=False,
                with_cluster_center=False,
                with_voxel_center=True,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                legacy=True
            ),
            'pts_middle_encoder': dict(
                type='PointPillarsScatter',
                in_channels=64,
                output_shape=(140, 88)
            ),
            'pts_backbone': dict(
                type='SECOND',
                in_channels=64,
                out_channels=[64, 128, 256],
                layer_nums=[3, 5, 5],
                layer_strides=[1, 2, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
            ),
            'pts_neck': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256],
                out_channels=[128, 128, 128],
                upsample_strides=[0.5, 1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True
            ),
            'occupancy_init': 0.01,
            'out_channels_pts': 80,
        }
        ################################################
        self.fuser_conf = {
            'img_dims': 80,
            'pts_dims': 80,
            'embed_dims': 128,
            'num_layers': 6,
            'num_heads': 4,
            'bev_shape': (128, 128),
        }
        ################################################
        self.head_conf = {
            'bev_backbone_conf': dict(
                type='ResNet',
                in_channels=128,
                depth=18,
                num_stages=3,
                strides=(1, 2, 2),
                dilations=(1, 1, 1),
                out_indices=[0, 1, 2],
                norm_eval=False,
                base_channels=160,
            ),
            'bev_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[128, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64]
            ),
            'tasks': [
                dict(num_class=1, class_names=['car']),
                dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
            ],
            'common_heads': dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            'bbox_coder': dict(
                type='CenterPointBBoxCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                code_size=9,
            ),
            'train_cfg': dict(
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                grid_size=[512, 512, 1],
                voxel_size=[0.2, 0.2, 8],
                out_size_factor=4,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            'test_cfg': dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=200,
                nms_thr=0.2,
            ),
            'in_channels': 256,  # Equal to bev_neck output_channels.
            'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
            'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            'gaussian_overlap': 0.1,
            'min_radius': 2,
        }
        ################################################
        self.key_idxes = [-2, -4, -6]
        self.model = CameraRadarNetDet(self.backbone_img_conf,
                                       self.backbone_pts_conf,
                                       self.fuser_conf,
                                       self.head_conf)

    def forward(self, sweep_imgs, mats, is_train=False, **inputs):
        return self.model(sweep_imgs, mats, sweep_ptss=inputs['pts_pv'], is_train=is_train)

    def training_step(self, batch):
        if self.global_rank == 0:
            for pg in self.trainer.optimizers[0].param_groups:
                self.log('learning_rate', pg["lr"])

        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            if self.return_radar_pv:
                pts_pv = pts_pv.cuda()
            gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
            gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
        preds, depth_preds = self(sweep_imgs, mats,
                                  pts_pv=pts_pv,
                                  is_train=True)
        targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...].contiguous()
        loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=3.)

        self.log('train/detection', loss_detection)
        self.log('train/heatmap', loss_heatmap)
        self.log('train/bbox', loss_bbox)
        self.log('train/depth', loss_depth)
        return loss_detection + loss_depth

    def validation_epoch_end(self, validation_step_outputs):
        detection_losses = list()
        heatmap_losses = list()
        bbox_losses = list()
        depth_losses = list()
        for validation_step_output in validation_step_outputs:
            detection_losses.append(validation_step_output[0])
            heatmap_losses.append(validation_step_output[1])
            bbox_losses.append(validation_step_output[2])
            depth_losses.append(validation_step_output[3])
        synchronize()

        self.log('val/detection', torch.mean(torch.stack(detection_losses)), on_epoch=True)
        self.log('val/heatmap', torch.mean(torch.stack(heatmap_losses)), on_epoch=True)
        self.log('val/bbox', torch.mean(torch.stack(bbox_losses)), on_epoch=True)
        self.log('val/depth', torch.mean(torch.stack(depth_losses)), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            if self.return_radar_pv:
                pts_pv = pts_pv.cuda()
            gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
            gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
        with torch.no_grad():
            preds, depth_preds = self(sweep_imgs, mats,
                                      pts_pv=pts_pv,
                                      is_train=True)

            targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
            loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

            if len(depth_labels.shape) == 5:
                # only key-frame will calculate depth loss
                depth_labels = depth_labels[:, 0, ...].contiguous()
            loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=3.)
        return loss_detection, loss_heatmap, loss_bbox, loss_depth


if __name__ == '__main__':
    run_cli(CRNLightningModel,
            'det/CRN_r50_256x704_128x128_4key')

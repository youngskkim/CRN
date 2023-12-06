import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models.backbones.resnet import BasicBlock
from .base_lss_fpn import BaseLSSFPN, Mlp, SELayer

from ops.average_voxel_pooling_v2 import average_voxel_pooling

__all__ = ['RVTLSSFPN']


class ViewAggregation(nn.Module):
    """
    Aggregate frustum view features transformed by depth distribution / radar occupancy
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ViewAggregation, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)
        return x


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels,
                 camera_aware=True):
        super(DepthNet, self).__init__()
        self.camera_aware = camera_aware

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        if self.camera_aware:
            self.bn = nn.BatchNorm1d(27)
            self.depth_mlp = Mlp(27, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_mlp = Mlp(27, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        x = self.reduce_conv(x)

        if self.camera_aware:
            intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
            batch_size = intrins.shape[0]
            num_cams = intrins.shape[2]
            ida = mats_dict['ida_mats'][:, 0:1, ...]
            sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
            bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                            4).repeat(1, 1, num_cams, 1, 1)
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            intrins[:, 0:1, ..., 0, 0],
                            intrins[:, 0:1, ..., 1, 1],
                            intrins[:, 0:1, ..., 0, 2],
                            intrins[:, 0:1, ..., 1, 2],
                            ida[:, 0:1, ..., 0, 0],
                            ida[:, 0:1, ..., 0, 1],
                            ida[:, 0:1, ..., 0, 3],
                            ida[:, 0:1, ..., 1, 0],
                            ida[:, 0:1, ..., 1, 1],
                            ida[:, 0:1, ..., 1, 3],
                            bda[:, 0:1, ..., 0, 0],
                            bda[:, 0:1, ..., 0, 1],
                            bda[:, 0:1, ..., 1, 0],
                            bda[:, 0:1, ..., 1, 1],
                            bda[:, 0:1, ..., 2, 2],
                        ],
                        dim=-1,
                    ),
                    sensor2ego.view(batch_size, 1, num_cams, -1),
                ],
                -1,
            )
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context_img = self.context_se(x, context_se)
            context = self.context_conv(context_img)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x)
            depth = self.depth_conv(x)

        return torch.cat([depth, context], dim=1)


class RVTLSSFPN(BaseLSSFPN):
    def __init__(self, **kwargs):
        super(RVTLSSFPN, self).__init__(**kwargs)

        self.register_buffer('frustum', self.create_frustum())
        self.z_bound = kwargs['z_bound']
        self.radar_view_transform = kwargs['radar_view_transform']
        self.camera_aware = kwargs['camera_aware']

        self.depth_net = self._configure_depth_net(kwargs['depth_net_conf'])
        self.view_aggregation_net = ViewAggregation(self.output_channels*2,
                                                    self.output_channels*2,
                                                    self.output_channels)

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            camera_aware=self.camera_aware
        )

    def get_geometry_collapsed(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                               z_min=-5., z_max=3.):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)).double()
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).double()
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points).half()
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z

    def _forward_view_aggregation_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.view_aggregation_net(img_feat_with_depth).view(
                n, h, c//2, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _split_batch_cam(self, feat, inv=False, num_cams=6):
        batch_size = feat.shape[0]
        if not inv:
            return feat.reshape(batch_size // num_cams, num_cams, *feat.shape[1:])
        else:
            return feat.reshape(batch_size * num_cams, *feat.shape[2:])

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              pts_context,
                              pts_occupancy,
                              return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego.
                intrin_mats(Tensor): Intrinsic matrix.
                ida_mats(Tensor): Transformation matrix for ida.
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera.
                bda_mat(Tensor): Rotation matrix for bda.
            ptss_context(Tensor): Input point context feature.
            ptss_occupancy(Tensor): Input point occupancy.
            return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        # extract image feature
        img_feats = self.get_cam_feats(sweep_imgs)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img_backbone'].append(t1.elapsed_time(t2))

        source_features = img_feats[:, 0, ...]
        source_features = self._split_batch_cam(source_features, inv=True, num_cams=num_cams)

        # predict image context feature, depth distribution
        depth_feature = self._forward_depth_net(
            source_features,
            mats_dict,
        )
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['img_dep'].append(t2.elapsed_time(t3))

        image_feature = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)]

        depth_occupancy = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)
        img_feat_with_depth = depth_occupancy.unsqueeze(1) * image_feature.unsqueeze(2)

        # calculate frustum grid within valid height
        geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None))

        geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True, num_cams=num_cams).unsqueeze(1)
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)

        if self.radar_view_transform:
            radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous()
            image_feature_collapsed = (image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2)
            img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)

            img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)
            img_context = self._forward_view_aggregation_net(img_context)
        else:
            img_context = img_feat_with_depth
        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['img_transform'].append(t3.elapsed_time(t4))

        img_context = self._split_batch_cam(img_context, num_cams=num_cams)
        img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous()

        pts_context = self._split_batch_cam(pts_context, num_cams=num_cams)
        pts_context = pts_context.unsqueeze(-2).permute(0, 1, 3, 4, 5, 2).contiguous()

        fused_context = torch.cat([img_context, pts_context], dim=-1)

        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0  # collapse z-axis
        geo_pos = torch.ones_like(geom_xyz)
        
        # sparse voxel pooling
        feature_map, _ = average_voxel_pooling(geom_xyz, fused_context.contiguous(), geo_pos,
                                               self.voxel_num.cuda())
        if self.times is not None:
            t5.record()
            torch.cuda.synchronize()
            self.times['img_pool'].append(t4.elapsed_time(t5))

        if return_depth:
            return feature_map.contiguous(), depth_feature[:, :self.depth_channels].softmax(1)
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                ptss_context,
                ptss_occupancy,
                times=None,
                return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            ptss_context(Tensor): Input point context feature with shape of
                (B * num_cameras, num_sweeps, C, D, W).
            ptss_occupancy(Tensor): Input point occupancy with shape of
                (B * num_cameras, num_sweeps, 1, D, W).
            times(Dict, optional): Inference time measurement.
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Return:
            Tensor: bev feature map.
        """
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            ptss_context[:, 0, ...] if ptss_context is not None else None,
            ptss_occupancy[:, 0, ...] if ptss_occupancy is not None else None,
            return_depth=return_depth)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if return_depth:
                return key_frame_res[0].unsqueeze(1), key_frame_res[1], self.times
            else:
                return key_frame_res.unsqueeze(1), self.times

        key_frame_feature = key_frame_res[0] if return_depth else key_frame_res
        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    ptss_context[:, sweep_index, ...] if ptss_context is not None else None,
                    ptss_occupancy[:, sweep_index, ...] if ptss_occupancy is not None else None,
                    return_depth=False)
                ret_feature_list.append(feature_map)

        if return_depth:
            return torch.stack(ret_feature_list, 1), key_frame_res[1], self.times
        else:
            return torch.stack(ret_feature_list, 1), self.times

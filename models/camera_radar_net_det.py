import mmcv

from models.base_bev_depth import BaseBEVDepth
from layers.backbones.rvt_lss_fpn import RVTLSSFPN
from layers.backbones.pts_backbone import PtsBackbone
from layers.fuser.multimodal_feature_aggregation import MFAFuser
from layers.heads.bev_depth_head_det import BEVDepthHead

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

__all__ = ['CameraRadarNetDet']


class CameraRadarNetDet(BaseBEVDepth):
    """Source code of `CRN`, `https://arxiv.org/abs/2304.00670`.

    Args:
        backbone_img_conf (dict): Config of image backbone.
        backbone_pts_conf (dict): Config of point backbone.
        fuser_conf (dict): Config of BEV feature fuser.
        head_conf (dict): Config of head.
    """

    def __init__(self, backbone_img_conf, backbone_pts_conf, fuser_conf, head_conf):
        super(BaseBEVDepth, self).__init__()
        self.backbone_img = RVTLSSFPN(**backbone_img_conf)
        self.backbone_pts = PtsBackbone(**backbone_pts_conf)
        self.fuser = MFAFuser(**fuser_conf)
        self.head = BEVDepthHead(**head_conf)

        self.radar_view_transform = backbone_img_conf['radar_view_transform']

        # inference time measurement
        self.idx = 0
        self.times_dict = {
            'img': [],
            'img_backbone': [],
            'img_dep': [],
            'img_transform': [],
            'img_pool': [],

            'pts': [],
            'pts_voxelize': [],
            'pts_backbone': [],
            'pts_head': [],

            'fusion': [],
            'fusion_pre': [],
            'fusion_layer': [],
            'fusion_post': [],

            'head': [],
            'head_backbone': [],
            'head_head': [],
        }

    def forward(self,
                sweep_imgs,
                mats_dict,
                sweep_ptss=None,
                is_train=False
                ):
        """Forward function for BEVDepth

        Args:
            sweep_imgs (Tensor): Input images.
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
            sweep_ptss (Tensor): Input points.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if is_train:
            self.time = None

            ptss_context, ptss_occupancy, _ = self.backbone_pts(sweep_ptss)
            feats, depth, _ = self.backbone_img(sweep_imgs,
                                                mats_dict,
                                                ptss_context,
                                                ptss_occupancy,
                                                return_depth=True)
            fused, _ = self.fuser(feats)
            preds, _ = self.head(fused)
            return preds, depth
        else:
            if self.idx < 100:  # skip few iterations for warmup
                self.times = None
            elif self.idx == 100:
                self.times = self.times_dict

            ptss_context, ptss_occupancy, self.times = self.backbone_pts(sweep_ptss,
                                                                         times=self.times)
            feats, self.times = self.backbone_img(sweep_imgs,
                                                  mats_dict,
                                                  ptss_context,
                                                  ptss_occupancy,
                                                  times=self.times)
            fused, self.times = self.fuser(feats, times=self.times)
            preds, self.times = self.head(fused, times=self.times)

            if self.idx == 1000:
                time_mean = {}
                for k, v in self.times.items():
                    time_mean[k] = sum(v) / len(v)
                print('img: %.2f' % time_mean['img'])
                print('  img_backbone: %.2f' % time_mean['img_backbone'])
                print('  img_dep: %.2f' % time_mean['img_dep'])
                print('  img_transform: %.2f' % time_mean['img_transform'])
                print('  img_pool: %.2f' % time_mean['img_pool'])
                print('pts: %.2f' % time_mean['pts'])
                print('  pts_voxelize: %.2f' % time_mean['pts_voxelize'])
                print('  pts_backbone: %.2f' % time_mean['pts_backbone'])
                print('  pts_head: %.2f' % time_mean['pts_head'])
                print('fusion: %.2f' % time_mean['fusion'])
                print('  fusion_pre: %.2f' % time_mean['fusion_pre'])
                print('  fusion_layer: %.2f' % time_mean['fusion_layer'])
                print('  fusion_post: %.2f' % time_mean['fusion_post'])
                print('head: %.2f' % time_mean['head'])
                print('  head_backbone: %.2f' % time_mean['head_backbone'])
                print('  head_head: %.2f' % time_mean['head_head'])
                total = time_mean['pts'] + time_mean['img'] + time_mean['fusion'] + time_mean['head']
                print('total: %.2f' % total)
                print(' ')
                print('FPS: %.2f' % (1000/total))

            self.idx += 1
            return preds

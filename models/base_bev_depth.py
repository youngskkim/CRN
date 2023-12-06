import mmcv
import torch
from torch import nn

from layers.backbones.base_lss_fpn import BaseLSSFPN
from layers.heads.bev_depth_head_det import BEVDepthHead

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

__all__ = ['BaseBEVDepth']


class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
    """

    def __init__(self, backbone_conf, head_conf):
        super(BaseBEVDepth, self).__init__()
        self.backbone_img = BaseLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)

        # for inference time measurement
        self.idx = 0
        self.times_dict = {
            'img': [],
            'img_backbone': [],
            'img_dep': [],
            'img_transform': [],
            'img_pool': [],

            'head': [],
            'head_backbone': [],
            'head_head': [],
        }

    def forward(self,
                sweep_imgs,
                mats_dict,
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

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if is_train:
            self.time = None

            x, depth, _ = self.backbone_img(sweep_imgs, mats_dict,
                                            is_return_depth=True)
            preds, _ = self.head(x)
            return preds, depth
        else:
            if self.idx < 100:  # skip few iterations for warmup
                self.times = None
            elif self.idx == 100:
                self.times = self.times_dict

            x, self.times = self.backbone_img(sweep_imgs, mats_dict,
                                              times=self.times)
            preds, self.times = self.head(x, times=self.times)

            if self.idx == 1000:
                time_mean = {}
                for k, v in self.times.items():
                    time_mean[k] = sum(v) / len(v)
                print('img: %.2f' % time_mean['img'])
                print('  img_backbone: %.2f' % time_mean['img_backbone'])
                print('  img_dep: %.2f' % time_mean['img_dep'])
                print('  img_transform: %.2f' % time_mean['img_transform'])
                print('  img_pool: %.2f' % time_mean['img_pool'])
                print('head: %.2f' % time_mean['head'])
                print('  head_backbone: %.2f' % time_mean['head_backbone'])
                print('  head_head: %.2f' % time_mean['head_head'])
                total = time_mean['img'] + time_mean['head']
                print('total: %.2f' % total)
                print(' ')
                print('FPS: %.2f' % (1000/total))

            self.idx += 1
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)

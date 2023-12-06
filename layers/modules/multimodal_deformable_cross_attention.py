# Based on https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
# ---------------------------------------------
#  Modified by Youngseok Kim
# ---------------------------------------------

import warnings
import math
import torch
import torch.nn as nn

from mmcv.cnn import xavier_init, constant_init
from mmcv.runner.base_module import BaseModule

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


class DeformableCrossAttention(BaseModule):
    """Multi-modal Feature Aggregation module used in CRN based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        img_dims (int): The embedding dimension of image feature map.
            Default: 128.
        pts_dims (int): The embedding dimension of radar feature map.
            Default: 128.
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 4.
        num_modalities (int): The number of feature map used in
            Attention. Default: 2.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_dims=128,
                 pts_dims=128,
                 embed_dims=256,
                 num_heads=4,
                 num_modalities=2,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step

        self.img_dims = img_dims
        self.pts_dims = pts_dims

        self.embed_dims = embed_dims
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims,
                                          num_modalities * num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_modalities * num_heads * num_points)
        self.value_proj_img = nn.Linear(img_dims, embed_dims)
        self.value_proj_pts = nn.Linear(pts_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_modalities, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj_img, distribution='uniform', bias=0.)
        xavier_init(self.value_proj_pts, distribution='uniform', bias=0.)

        self._is_init = True

    def forward(self,
                queries,
                value_img,
                value_pts,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        """Forward Function of DeformableCrossAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key, img_dims/pts_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key, img_dims/pts_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_modalities, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_modalities, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_modalities, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_modalities, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if identity is None:
            identity = queries
        if query_pos is not None:
            queries = queries + query_pos

        bs, num_query, _ = queries.shape
        _, num_value_img, _ = value_img.shape
        _, num_value_pts, _ = value_pts.shape

        value_img = self.value_proj_img(value_img)
        value_img = value_img.reshape(bs, num_value_img, self.num_heads, -1)
        value_pts = self.value_proj_pts(value_pts)
        value_pts = value_pts.reshape(bs, num_value_pts, self.num_heads, -1)
        value = torch.cat([value_img, value_pts], dim=1)

        sampling_offsets = self.sampling_offsets(queries).view(
            bs, num_query, self.num_heads, self.num_modalities, self.num_points, 2)
        attention_weights = self.attention_weights(queries).view(
            bs, num_query, self.num_heads, self.num_modalities * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_modalities,
                                                   self.num_points)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, :, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)

        return self.dropout(output) + identity


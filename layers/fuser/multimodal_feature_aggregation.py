import torch
from torch import nn

from mmcv.runner.base_module import ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmcv.runner import auto_fp16

from ..modules.multimodal_deformable_cross_attention import DeformableCrossAttention


class MFAFuser(nn.Module):
    def __init__(self, num_sweeps=4, img_dims=80, pts_dims=128, embed_dims=256,
                 num_layers=6, num_heads=4, bev_shape=(128, 128)):
        super(MFAFuser, self).__init__()

        self.num_modalities = 2
        self.use_cams_embeds = False

        self.num_heads = num_heads

        self.img_dims = img_dims
        self.pts_dims = pts_dims
        self.embed_dims = embed_dims
        _pos_dim_ = self.embed_dims//2
        _ffn_dim_ = self.embed_dims*2

        self.norm_img = build_norm_layer(dict(type='LN'), img_dims)[1]
        self.norm_pts = build_norm_layer(dict(type='LN'), pts_dims)[1]
        self.input_proj = nn.Linear(img_dims + pts_dims, self.embed_dims)

        self.bev_h, self.bev_w = bev_shape

        self.positional_encoding = build_positional_encoding(
            dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=self.bev_h,
                col_num_embed=self.bev_w,
            ),
        )
        self.register_buffer('ref_2d', self.get_reference_points(self.bev_h, self.bev_w))

        ffn_cfgs = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=_ffn_dim_,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        norm_cfgs = dict(type='LN')

        self.ffn_layers = ModuleList()
        for _ in range(num_layers):
            self.ffn_layers.append(
                build_feedforward_network(ffn_cfgs)
            )
        self.norm_layers1 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers1.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.norm_layers2 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers2.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.attn_layers = ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                DeformableCrossAttention(
                    img_dims=self.img_dims,
                    pts_dims=self.pts_dims,
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    num_modalities=self.num_modalities,
                    num_points=4
                ),
            )

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(embed_dims*num_sweeps,
                      embed_dims,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableCrossAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @staticmethod
    def get_reference_points(H, W, dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.unsqueeze(2).unsqueeze(3)
        return ref_2d

    @auto_fp16(apply_to=('feat_img', 'feat_pts'))
    def _forward_single_sweep(self, feat_img, feat_pts):
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        bs = feat_img.shape[0]
        ref_2d_stack = self.ref_2d.repeat(bs, 1, 1, self.num_modalities, 1)

        feat_img = self.norm_img(feat_img.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        feat_pts = self.norm_pts(feat_pts.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        feat_flatten = []
        spatial_shapes = []
        for feat in [feat_img, feat_pts]:
            _, _, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2, 1).contiguous()  # [bs, num_cam, c, dw] -> [num_cam, bs, dw, c]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_img.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        bev_queries = torch.cat(feat_flatten, -1)
        bev_queries = self.input_proj(bev_queries)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(feat_img.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(feat_img.dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()

        feat_img = feat_flatten[0]
        feat_pts = feat_flatten[1]
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['fusion_pre'].append(t1.elapsed_time(t2))

        for attn_layer, ffn_layer, norm_layer1, norm_layer2 in \
            zip(self.attn_layers, self.ffn_layers, self.norm_layers1, self.norm_layers2):
            # post norm
            bev_queries = attn_layer(
                bev_queries,
                feat_img,
                feat_pts,
                identity=None,
                query_pos=bev_pos,
                reference_points=ref_2d_stack,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                )
            bev_queries = norm_layer1(bev_queries)
            bev_queries = ffn_layer(bev_queries, identity=None)
            bev_queries = norm_layer2(bev_queries)
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['fusion_layer'].append(t2.elapsed_time(t3))

        output = bev_queries.permute(0, 2, 1).contiguous().reshape(bs, self.embed_dims, h, w)
        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['fusion_post'].append(t3.elapsed_time(t4))

        return output

    def forward(self, feats, times=None):
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        num_sweeps = feats.shape[1]
        key_frame_res = self._forward_single_sweep(
            feats[:, 0, :self.img_dims],
            feats[:, 0, self.img_dims:self.img_dims+self.pts_dims]
        )
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['fusion'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            return key_frame_res, self.times

        ret_feature_list = [key_frame_res]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    feats[:, sweep_index, :self.img_dims],
                    feats[:, sweep_index, self.img_dims:self.img_dims+self.pts_dims])
                ret_feature_list.append(feature_map)

        return self.reduce_conv(torch.cat(ret_feature_list, 1)).float(), self.times
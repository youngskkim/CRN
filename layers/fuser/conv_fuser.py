import torch
from torch import nn
from torch.cuda.amp import autocast

from mmcv.runner.base_module import ModuleList

class ConvFuser(nn.Module):
    def __init__(self, num_sweeps=4, img_dims=80, pts_dims=80, embed_dims=128, num_layers=3, **kwargs):
        super(ConvFuser, self).__init__()

        self.conv_layers = ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = img_dims + pts_dims
            else:
                input_dims = embed_dims
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dims,
                              embed_dims,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(embed_dims),
                    nn.ReLU(inplace=True),
                )
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
            nn.Conv2d(embed_dims,
                      embed_dims,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.idx = 0

    @autocast(False)
    def forward(self, x, times=None):
        num_sweeps = x.shape[1]

        self.times = times
        if self.idx > 100:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        key_frame_res = x[:, 0, ...]
        for conv_layer in self.conv_layers:
            key_frame_res = conv_layer(key_frame_res)

        if self.idx > 100:
            t2.record()
            torch.cuda.synchronize()
            self.times['fusion'].append(t1.elapsed_time(t2))

        if times is not None: self.idx += 1
        if num_sweeps == 1:
            return self.reduce_conv(key_frame_res), self.times

        ret_feature_list = [key_frame_res]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = x[:, sweep_index, ...]
                for conv_layer in self.conv_layers:
                    feature_map = conv_layer(feature_map)
                ret_feature_list.append(feature_map)

        return self.reduce_conv(torch.cat(ret_feature_list, 1)), self.times
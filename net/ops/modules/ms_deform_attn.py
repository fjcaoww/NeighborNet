# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        n_levels = int(n_levels)

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.value_head = nn.Conv2d(self.d_model//self.n_heads, 2, kernel_size=1, stride=1)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def neighbor_uncer(self, value):
        # value shape: n*m, d, h, w
        n, m, d, h, w = value.shape
        value = value.reshape(n*m, d, h, w)
        preds = self.value_head(value)
        preds = torch.softmax(preds, dim=1)
        preds = preds.reshape(n, m, 2, h, w)
        preds_ds = preds
        preds = torch.mean(preds, dim=1)    # shape:n, 2, h, w
        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
        return uncertainty, preds_ds

    def forward(self, mask_num, dis, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        sample = input_level_start_index

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        if sample > 1:    # CF不使用邻居数量正则化  1.5编码也不使用正则化
            index = mask_num.shape[1]
            mask_num_pre = mask_num[:, :index - 1, :, :]
            mask_threshold = mask_num[:, index - 1:index, :, :]
            mask_number = torch.ge(mask_num_pre, mask_threshold)
            mask_number = mask_number.float()
        else:
            # 利用这里的多头value,计算不确定性图   #######################
            un_value = value.permute(0, 2, 3, 1).view(N, self.n_heads, self.d_model // self.n_heads, input_spatial_shapes[:, 0], input_spatial_shapes[:, 1])
            uncertainty, preds_ds = self.neighbor_uncer(un_value)
            mask_uncer = 1 - torch.pow(uncertainty, 5)  # 取uncertainty的10次方, 再 不确定性越高,值越小

            # 计算mask_number, 用不确定性图修正
            index = mask_num.shape[1]
            mask_num_pre = mask_num[:, :index - 1, :, :]
            mask_threshold = mask_num[:, index - 1:index, :, :]
            mask_threshold = mask_threshold * mask_uncer  # 对于不确定性高的位置降低阈值,增加点的数量
            mask_number = torch.ge(mask_num_pre, mask_threshold)
            mask_number = mask_number.float()

        # 使用mask_number自适应sampling_offset和attention_weight
        if sample > 1:    # CF自适应邻居重写代码
            mask_number = mask_number.permute(0, 2, 3, 1)
            mask_num1 = mask_number.reshape(N, Len_q, self.n_heads * self.n_levels * self.n_points, 1)
            sampling_offsets = self.sampling_offsets(query)
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads * self.n_levels * self.n_points, 2)
            sampling_offsets = sampling_offsets * mask_num1
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

            mask_num2 = mask_number.reshape(N, Len_q, self.n_heads * self.n_levels * self.n_points)
            attention_weights = self.attention_weights(query)
            attention_weights = attention_weights.view(N, Len_q, self.n_heads * self.n_levels * self.n_points)
            attention_weights = attention_weights * mask_num2
            attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        else:    # 1.5可以共用
            mask_number = mask_number.permute(0, 2, 3, 1)
            mask_num1 = mask_number.view(N, Len_q, self.n_heads*self.n_levels*self.n_points, 1)
            # sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            sampling_offsets = self.sampling_offsets(query)
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads*self.n_levels*self.n_points, 2)
            sampling_offsets = sampling_offsets * mask_num1
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

            # attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            mask_num2 = mask_number.view(N, Len_q, self.n_heads*self.n_levels*self.n_points)
            attention_weights = self.attention_weights(query)
            attention_weights = attention_weights.view(N, Len_q, self.n_heads*self.n_levels*self.n_points)
            attention_weights = attention_weights * mask_num2
            attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # sampling_locations = reference_points[:, :, None, :, None, :] \
            #                      + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # reference_points_fj = reference_points[:, :, None, :, None, :]
            # offset_normalizer_fj = offset_normalizer[None, None, None, :, None, :]
            # nor_sampling = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = reference_points[:, :, None, :, None, :] + (sampling_offsets / offset_normalizer[None, None, None, :, None, :]) * dis
            # sampling_locations1 = reference_points[:, :, None, :, None, :] + (sampling_offsets / offset_normalizer[None, None, None, :, None, :])
            # sampling_locations2 = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        if sample > 1:
            return output
        else:
            return output, preds_ds





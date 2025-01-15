import copy

import numpy as np
from torch import nn
from mlp import build_mlps

from einops.layers.torch import Rearrange
from PatchingFlatten import PatchingLayer
from PatchingFlatten import newPatchingLayer
from PatchingFlatten import PatchingDifined
from PatchingFlatten import Flatten_Head
from PatchingFlatten import Flatten_Head1

import torch.nn.functional as F
import torch


class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)
        # self.motion_mlp2 = build_mlps(self.config.motion_mlp2)
        self.motion_mlp2 = build_mlps(self.config.motion_mlp2)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct,
                                          self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct,
                                           self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.patching_layer = newPatchingLayer(padding_patch='end', patch_len=25, stride=5, pad=0)
        # self.patching_layer = newPatchingLayer(padding_patch='end', patch_len=25, stride=6, pad=5)
        self.patching_layer1 = newPatchingLayer(padding_patch='end', patch_len=12, stride=3, pad=0)
        # self.patching_layer1 = PatchingDifined1()

        self.flatten = Flatten_Head(individual=False, n_vars=66, nf=300, target_window=50, head_dropout=0.)
        self.flatten1 = Flatten_Head(individual=False, n_vars=50, nf=570, target_window=66, head_dropout=0.)
        # self.flatten1 = Flatten_Head1(nf=150, target_window=66, head_dropout=0.)
        # self.flatten1 = Flatten_Head1()
        self.fc = nn.Linear(25, 50)
        self.fc1 = nn.Linear(12, 30)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)  # xavier初始化  每一层网络保证输入输出方差相同
        nn.init.constant_(self.motion_fc_out.bias, 0)  # 初始化整个矩阵为常数
        nn.init.xavier_uniform_(self.fc3.weight, gain=1e-8)
        nn.init.constant_(self.fc3.bias, 0)
        # nn.init.xavier_uniform_(self.fuse.weight, gain=1e-8)
        # nn.init.constant_(self.fuse.bias, 0)

    def forward(self, motion_input, xs):  # motion_input: [bs x 50 x 66]

        # xs = motion_input.clone()
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)  # motion_feats: [bs x 50 x 66]
            motion_feats = self.arr0(motion_feats)  # motion_feats: [bs x 66 x 50]
            xs = self.arr0(xs)
            xs = self.fc2(xs)
        xs = xs.permute(0, 2, 1)

        motion_feats1 = self.patching_layer(motion_feats)  # b,66,6,25

        motion_feats1 = self.fc(motion_feats1)  # b,66,10,50
        motion_feats = self.motion_mlp(motion_feats1)
        # motion_feats = self.se(motion_feats)
        motion_feats = self.flatten(motion_feats)  # b,66,50
        # xs1 = self.mlps(xs).permute(0, 2, 1)
        xs = self.patching_layer1(xs)
        xs = self.fc1(xs)
        xs = self.motion_mlp2(xs)
        # xs = self.mlp2(xs).permute(0, 2, 1)
        # xs = self.motion_mlp2(xs).permute(0, 2, 1)
        # xs = self.se1(xs)
        xs = self.flatten1(xs).permute(0, 2, 1)  # b,66,50//////////////////////
        # xs = self.flatten1(xs)                                   # b,50,66

        # As = torch.softmax(torch.matmul(xs.permute(0, 2, 1), motion_feats.permute(0, 2, 1)), dim=-1)          # b,66,66
        # xs = xs.permute(0, 2, 1) + torch.matmul(As, motion_feats)

        xs = self.fc3(xs)
        # output_incre = self.fc4(xs)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)  # motion_feats: [bs x 66 x 50]
            motion_feats = self.motion_fc_out(motion_feats)  # motion_feats: [bs x 50 x 66]
        # motion_feats = self.revin_layer(motion_feats, 'denorm')
        return motion_feats + xs.permute(0, 2, 1), xs.permute(0, 2, 1)


class Embedding_FC(nn.Module):
    def __init__(self, dim):
        super(Embedding_FC, self).__init__()
        self.fc = nn.Linear(50, 100)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock1(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, hf, dropout=0., dropout1=0.):
        super(MLPblock1, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout1)
        # self.gelu = nn.gule()
        self.ln = nn.LayerNorm(hf)

    def forward(self, x):
        x_ = self.ln(x)
        x_ = x_.permute(0, 1, 3, 2)
        # x_ = self.dropout(self.w_1(x).relu())
        x_ = self.w_2(self.dropout(self.w_1(x_).relu())).permute(0, 1, 3, 2)
        return x_ + x


class MLPblock2(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, hf, dropout=0.):
        super(MLPblock2, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.gelu = nn.gule()
        self.ln = nn.LayerNorm(hf)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_1.weight, gain=1e-8)
        nn.init.constant_(self.w_1.bias, 0)
        nn.init.xavier_uniform_(self.w_2.weight, gain=1e-8)
        nn.init.constant_(self.w_2.bias, 0)

    def forward(self, x):
        x_ = self.ln(x)
        # x_ = self.dropout(self.w_1(x).relu())
        x_ = self.w_2(self.dropout(self.w_1(x_).relu()))
        return x_ + x


# class CombinedNet(nn.Module):
#     def __init__(self, n_repeats):
#         super(CombinedNet, self).__init__()
#         self.n_repeats = n_repeats
#         self.module_list = nn.ModuleList([nn.Sequential(MLPblock1(6, 16, 25), MLPblock2(25, 64, 25)) for _ in range(n_repeats)])
#
#     def forward(self, x):
#         for module in self.module_list:
#             x = module(x)
#         return x
#
def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

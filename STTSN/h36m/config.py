
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 304

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'origin'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'lib'))

"""Data Dir and Weight Dir"""
# TODO

"""Dataset Config"""
C.h36m_anno_dir = osp.join(C.root_dir, 'data/h36m/')
C.motion = edict()

C.motion.h36m_input_length = 50
C.motion.h36m_input_length_dct = 50
C.motion.h36m_target_length_train = 10
C.motion.h36m_target_length_eval = 25
C.motion.dim = 66

C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False
## DC-Mixer tB
C.dcmixer_tB = edict()
C.dcmixer_tB.hidden_dim = 66                                         # MLP 的维度
C.dcmixer_tB.seq_len = C.motion.h36m_input_length_dct
C.dcmixer_tB.seq_len = 50                                           # temporal MLP 的维度
C.dcmixer_tB.num_layers = 48
C.dcmixer_tB.with_normalization = True
C.dcmixer_tB.spatial_fc_only = False
C.dcmixer_tB.norm_axis = 'spatial'
C.dcmixer_tB.pn = 6
## DC-Mixer sB
C.dcmixer_sB = edict()
C.dcmixer_sB.hidden_dim = 50                                         # MLP 的维度
C.dcmixer_sB.seq_len = C.motion.h36m_input_length_dct
C.dcmixer_sB.seq_len = 30                                           # temporal MLP 的维度
C.dcmixer_sB.num_layers = 48
C.dcmixer_sB.with_normalization = True
C.dcmixer_sB.spatial_fc_only = False
C.dcmixer_sB.norm_axis = 'spatial'
C.dcmixer_sB.pn = 19
## Motion Network FC In
C.dcmixer_fc_in = edict()
C.dcmixer_fc_in.in_features = C.motion.dim
C.dcmixer_fc_in.out_features = 66
C.dcmixer_fc_in.with_norm = False
C.dcmixer_fc_in.activation = 'relu'
C.dcmixer_fc_in.init_w_trunc_normal = False
C.dcmixer_fc_in.temporal_fc = False
## Motion Network FC Out
C.dcmixer_fc_out = edict()
C.dcmixer_fc_out.in_features = 66
C.dcmixer_fc_out.out_features = C.motion.dim
C.dcmixer_fc_out.with_norm = False
C.dcmixer_fc_out.activation = 'relu'
C.dcmixer_fc_out.init_w_trunc_normal = True
C.dcmixer_fc_out.temporal_fc = False

"""Train Config"""
C.batch_size = 128
C.num_workers = 8

C.cos_lr_max=1e-5
C.cos_lr_min=5e-8
C.cos_lr_total_iters=100000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 2000

if __name__ == '__main__':
    print(config.decoder.dcmixer_tB)

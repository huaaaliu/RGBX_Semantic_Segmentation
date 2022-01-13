import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
C.abs_dir = osp.realpath(".")
C.log_dir = osp.abspath('log')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Dataset Path"""
C.dataset_path = osp.join(C.root_dir, 'datasets', 'rgbt')
C.img_root_folder = osp.join(C.dataset_path, 'rgb')
C.gt_root_folder = osp.join(C.dataset_path, 'labels')
C.aolp_tri_root_folder = osp.join(C.dataset_path, 'polar')
C.polar_root_folder = osp.join(C.dataset_path, 'polar')
C.polar_root_folder = osp.join(C.dataset_path, 'polar')
C.polar_root_folder = osp.join(C.dataset_path, 'polar')
C.polar_root_folder = osp.join(C.dataset_path, 'polar')
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

"""Image Config"""
C.num_classes = 9
C.background = 255
# Statistik on train set
C.image_mean = np.array([0.22156, 0.25873, 0.23003])
C.image_std = np.array([0.16734, 0.16907, 0.16801])
C.aolp_tri_mean = np.array([0.48459, 0.48487, 0.48329])
C.aolp_tri_std = np.array([0.26167, 0.26139, 0.26288])
C.dolp_tri_mean = np.array([0.15018, 0.14808, 0.16881])
C.dolp_tri_std = np.array([0.12657, 0.12505, 0.13478])
C.aolp_mean = np.array([0.48442, 0.48442, 0.48442])
C.aolp_std = np.array([0.27697, 0.27697, 0.27697])
C.dolp_mean = np.array([0.14472, 0.14472, 0.14472])
C.dolp_std = np.array([0.12005, 0.12005, 0.12005])
C.polar_mean = np.array([0.39541])
C.polar_std = np.array([0.07578])
C.image_height = 480
C.image_width = 640
C.num_train_imgs = 784
C.num_eval_imgs = 393 #205 day #188 night

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2' # Remember change the path below.
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512 # valid for MLP decoder
C.optimizer = 'AdamW' #'SGDM'# 

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 12
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_base_size = [480, 640] 
C.eval_crop_size = [480, 640]
"""Display Config"""
C.save_start_epoch = 150
C.snapshot_iter = 25
C.record_info_iter = 200
C.display_iter = 200


if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
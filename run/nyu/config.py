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
C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDepthv2')
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.hha_root_folder = osp.join(C.dataset_path, 'HHA')
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

"""Image Config"""
C.num_classes = 40
C.background = 255
# Statistik on train set
C.image_mean = np.array([0.485, 0.456, 0.406]) # np.array([0.48474, 0.41623, 0.39792])
C.image_std = np.array([0.229, 0.224, 0.225])  # np.array([0.26400, 0.27250, 0.28229])
C.hha_mean = np.array([0.485, 0.456, 0.406]) # np.array([0.52234, 0.44826, 0.43878])
C.hha_std = np.array([0.229, 0.224, 0.225]) # np.array([0.17047, 0.24475, 0.13624])
C.image_height = 480
C.image_width = 640
C.num_train_imgs = 795
C.num_eval_imgs = 654

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b5' # Remember change the path below.
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b5.pth'
C.decoder = 'MLPDecoder' #'base' #'UPernet'# 
C.decoder_embed_dim = 512 # valid for MLP decoder
C.optimizer = 'AdamW' #'SGDM'# 

"""Train Config"""
C.lr = 6e-5 # 0.01 #
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01 # 5e-4 #
C.batch_size = 8
C.nepochs = 400
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
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
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
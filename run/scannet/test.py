import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from scannet import ScanNet
from models.builder import EncoderDecoder as segmodel
from dataloader import ValPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        depth = data['depth_img']
        name = data['fn'][20:]
        pred = self.sliding_eval_rgbX(img, depth, config.eval_crop_size, config.eval_stride_rate, device)

        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn = name + '.png'
            def demap_label_image(image):
                mapped = np.zeros(image.shape)
                for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
                    mapped[image==i] = x
                return mapped.astype(np.uint8)
            pred = demap_label_image(pred)
            pred = cv2.resize(pred, (1296, 968), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        return None

    def compute_metric(self, results):

        return "Finised!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default='./test_cmx')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'root': config.dataset_path,
                    'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'depth_root': config.depth_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = ScanNet(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, 0, 1, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 False)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

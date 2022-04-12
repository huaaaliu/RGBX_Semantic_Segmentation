import cv2
import torch
import numpy as np
from torch.utils import data
import torch.nn.functional as F
import random
from config import config
import utils.rgbx_transforms as rgbx_transforms

def random_mirror(img, gt, depth):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)
        depth = cv2.flip(depth, 1)

    return img, gt, depth


class TrainPre(object):
    def __init__(self, cfg):
        self.img_mean = cfg.image_mean
        self.img_std = cfg.image_std

    def __call__(self, img, gt, depth):
        img, gt, depth = random_mirror(img, gt, depth)
        if config.train_scale_array is not None:
            img, gt, depth, _ = rgbx_transforms.random_scale_rgbx(img, gt, depth, config.train_scale_array)

        img = rgbx_transforms.normalize(img, self.img_mean, self.img_std)
        depth = rgbx_transforms.normalize(depth, 0, 1)  #self.img_mean, self.img_std) #
        crop_size = (config.image_height, config.image_width)
        if img.shape[:2][0] != crop_size[0] or img.shape[:2][1] != crop_size[1]:
            crop_pos = rgbx_transforms.generate_random_crop_pos(img.shape[:2], crop_size)
            p_img, _ = rgbx_transforms.random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
            p_gt, _ = rgbx_transforms.random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            p_depth, _ = rgbx_transforms.random_crop_pad_to_shape(depth, crop_pos, crop_size, 0)
        else:
            p_img = img
            p_gt = gt
            p_depth = depth
        
        p_img = p_img.transpose(2, 0, 1)
        p_depth = p_depth.transpose(2, 0, 1)

        extra_dict = {'depth_img': p_depth}
        
        return p_img, p_gt, extra_dict

class ValPre(object):
    def __call__(self, img, gt, depth):
        extra_dict = {'depth_img': depth}
        return img, gt, extra_dict

def get_train_loader(engine, dataset):
    data_setting = {'root': config.dataset_path,
                    'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'depth_root':config.depth_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(img, gt, ther):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)
        ther = cv2.flip(ther, 1)
    return img, gt, ther

def random_scale(img, gt, ther, scales):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ther = cv2.resize(ther, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return img, gt, ther, scale

class TrainPre(object):
    def __init__(self, img_mean, img_std, ther_mean, ther_std):
        self.img_mean = img_mean
        self.img_std = img_std
        self.ther_mean = ther_mean
        self.ther_std = ther_std

    def __call__(self, img, gt, ther):
        img, gt, ther = random_mirror(img, gt, ther)
        if config.train_scale_array is not None:
            img, gt, ther, scale = random_scale(img, gt, ther, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)
        ther = normalize(ther, self.ther_mean, self.ther_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_ther, _ = random_crop_pad_to_shape(ther, crop_pos, crop_size, 0)

        p_img = p_img.transpose(2, 0, 1)
        p_ther = p_ther.transpose(2, 0, 1)

        extra_dict = {'ther_img': p_ther}
        
        return p_img, p_gt, extra_dict

class ValPre(object):
    def __call__(self, img, gt, ther):
        extra_dict = {'ther_img': ther}
        return img, gt, extra_dict

def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'ther_root':config.ther_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std, config.ther_mean, config.ther_std)

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

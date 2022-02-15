import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.rgbx_transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(img, gt, hha):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)
        hha = cv2.flip(hha, 1)

    return img, gt, hha

def random_scale(img, gt, hha, scales):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    hha = cv2.resize(hha, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return img, gt, hha, scale

class TrainPre(object):
    def __init__(self, img_mean, img_std, hha_mean, hha_std):
        self.img_mean = img_mean
        self.img_std = img_std
        self.hha_mean = hha_mean
        self.hha_std = hha_std

    def __call__(self, img, gt, hha):
        img, gt, hha = random_mirror(img, gt, hha)
        if config.train_scale_array is not None:
            img, gt, hha, scale = random_scale(img, gt, hha, config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)
        hha = normalize(hha, self.hha_mean, self.hha_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_hha, _ = random_crop_pad_to_shape(hha, crop_pos, crop_size, 0)

        p_img = p_img.transpose(2, 0, 1)
        p_hha = p_hha.transpose(2, 0, 1)

        extra_dict = {'hha_img': p_hha}
        
        return p_img, p_gt, extra_dict

class ValPre(object):
    def __call__(self, img, gt, hha):
        extra_dict = {'hha_img': hha}
        return img, gt, extra_dict

def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root':config.hha_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std, config.hha_mean, config.hha_std)

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

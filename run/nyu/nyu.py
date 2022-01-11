# encoding: utf-8
import numpy as np
import torch
from engine.BaseDataset import BaseDataset
import os
import cv2

class NYUv2(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(NYUv2, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._hha_path = setting['hha_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path ,names[0])
        gt_path = os.path.join(self._gt_path ,names[1])
        item_name = names[1].split("/")[-1].split(".")[0]
        hha_path = os.path.join(self._hha_path, item_name + '.jpg')#'.npy')#

        img, gt, hha = self._fetch_data(img_path, gt_path, hha_path)

        img = img[:, :, ::-1]
        hha = hha[:, :, ::-1]
        gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255
        
        
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt, hha)

        if self._split_name == 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name), n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, hha_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hha =  self._open_image(hha_path)
        return img, gt, hha

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @classmethod
    def get_class_names(*args):
        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name

import numpy as np
import torch
from engine.BaseDataset import BaseDataset
import os
import cv2

class SID(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(SID, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self.root = setting['root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length

        self.preprocess = preprocess

    def __getitem__(self, index):
        if self._file_length is not None:
            area, name = self._construct_new_file_names(self._file_length)[index]
        else:
            area, name = self._file_names[index]

        img_path = os.path.join(self.root, area, 'image', name + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hha_path = os.path.join(self.root, area, 'hha', name + '.png')
        hha = cv2.imread(hha_path, cv2.COLOR_BGR2RGB)
        
        gt_path = os.path.join(self.root, area, 'label', name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt -= 1       # 0->255

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
            
        output_dict = dict(data=img, label=gt, fn=str(name), n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)
        
        return output_dict

    def _get_file_names(self, split_name, train_extra=False):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            area, img_name = self._process_item_names(item)
            file_names.append([area, img_name])

        if train_extra:
            file_names2 = []
            source2 = self._train_source.replace('train', 'train_extra')
            with open(source2) as f:
                files2 = f.readlines()

            for item in files2:
                img_name, gt_name = self._process_item_names(item)
                file_names2.append([img_name, gt_name])

            return file_names, file_names2

        return file_names
    
    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        item = item.split()
        area = item[0]
        img_name = item[1]

        return area, img_name

    
    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 14
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
        return ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name
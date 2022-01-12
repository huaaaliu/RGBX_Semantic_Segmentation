# RGBX_Semantic_Segmentation



**！！！代码更新： 网络结构最后改了一遍，然后segformer的pretrain要重新下载一下。
配置： Segformer用 MLPDecoder（dim=512/768）， Swin 用‘UPernet’ dim=512。**

**Install dependencies.**

   * Python + numpy
   * Pytorch 1.7.0 + cudnn
   * CUDA 10.2
   * easydict 1.9
   * opencv-python 4.5.0
   * timm 0.4.12


## Data preparation


#### How to generate HHA maps?

If you want to generate HHA maps from Depth maps, please refer to [https://github.com/charlesCXK/Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python).

#### Pretrained models

Download the pretrained segformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing).

​

### Training

Training on NYU Depth V2:

Single GPU training:
```shell
$ cd ./model/mmt.nyu
$ CUDA_VISIBLE_DEVICES="GPU IDs" python train.py -d=0
```

Multi GPU training:
```shell
$ cd ./model/mmt.nyu
$ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py
```

- The tensorboard file is saved in `log/tb/` directory.



## Acknowledgement

Thanks [TorchSeg](https://github.com/ycszen/TorchSeg) and [SA-Gate](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch) for their excellent project!

​

## TODO

- [ ] More encoders.
- [ ] Different datasets.
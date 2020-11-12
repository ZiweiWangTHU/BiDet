# BiDet
This is the official pytorch implementation for paper: [*BiDet: An Efficient Binarized Object Detector*](https://arxiv.org/abs/2003.03961), which is accepted by CVPR2020. The code contains training and testing two binarized object detectors, SSD300 and Faster R-CNN, using our BiDet method on two datasets, PASCAL VOC and Microsoft COCO 2014.

## Quick Start
### Prerequisites
- python 3.5+
- pytorch 1.0+
- other packages include numpy, cv2, matplotlib, pillow, cython, cffi, msgpack, easydict, pyyaml

**Note: **as [this issue](https://github.com/ZiweiWangTHU/BiDet/issues/17#issuecomment-725796486) pointed out, this repo is not compatible with PyTorch 1.7.0+. You can follow that instruction to modify the code and make it runnable using PyTorch 1.7.

### Dataset Preparation
We conduct experiments on PASCAL VOC and Microsoft COCO 2014 datasets.  
#### PASCAL VOC
We train our model on the VOC 0712 trainval sets and test it on the VOC 07 test set. For downloading, just run:  

```shell
sh data/scripts/VOC2007.sh # <directory>
sh data/scripts/VOC2012.sh # <directory>
```

Please specify a path to download your data in, or the default path is ~/data/.  
#### COCO
We train our model on the COCO 2014 trainval35k subset and evaluate it on minival5k. For downloading, just run:  

```shell
sh data/scripts/COCO2014.sh
```

Also, you can specify a path to save the data.  

After downloading both datasets, please modify file faster_rcnn/lib/datasets/factory.py line 24 and file faster_rcnn/lib/datasets/coco.py line 36 by replacing path/to/dataset with your voc and coco dataset path respectively.  

### Pretrained Backbone
The backbones for our BiDet-SSD300 and BiDet-Faster R-CNN are VGG16 and Resnet-18. We pretrain them on the ImageNet dataset. You can download the pretrained weights on: [VGG16](https://drive.google.com/file/d/1K0hJasYqeUnz82FcB2XnCca8vzsLQcBv/view?usp=sharing) and [ResNet18](https://drive.google.com/file/d/1SB5oPbGX-MBwjv0QHBbgVRKVpb-3VY00/view?usp=sharing). After downloading them from Google Drive, please put them in ssd/pretrain and faster_rcnn/pretrain respectively.  

### Training and Testing
Assume you've finished all steps above, you can start using the code easily.  

#### SSD
For training SSD, just run:  

```shell
$ python ssd/train_bidet_ssd.py --dataset='VOC/COCO' --data_root='path/to/dataset' --basenet='path/to/pretrain_backbone'
```

For testing on VOC, just run:  

```shell
$ python ssd/eval_voc.py --weight_path='path/to/weight' --voc_root='path/to/voc'
```

For testing on COCO, just run:  

```shell
$ python ssd/eval_coco.py --weight_path='path/to/weight' --coco_root='path/to/coco'
```

#### Faster R-CNN
First you need to compile the cuda implementation for RoIPooling, RoIAlign and NMS. Just do:  

```shell
cd faster_rcnn/lib
python setup.py build develop
```


For training Faster R-CNN, just run:  

```shell
$ python faster_rcnn/trainval_net.py --dataset='voc/coco' --data_root='path/to/dataset' --basenet='path/to/pretrain_backbone'
```

For testing, run:  

```shell
$ python test_net.py --dataset='voc/coco' --checkpoint='path/to/weight'
```

## Citation
Please cite our paper if you find it useful in your research:

```
@inproceedings{wang2020bidet,
  title={BiDet: An Efficient Binarized Object Detector},
  author={Wang, Ziwei and Wu, Ziyi and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2049--2058},
  year={2020}
}
```

## Contact
If you have any questions about the code, please contact Ziyi Wu wuzy17@mails.tsinghua.edu.cn


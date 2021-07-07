# BiDet
This is the official pytorch implementation for paper: [*BiDet: An Efficient Binarized Object Detector*](https://arxiv.org/abs/2003.03961), which is accepted by CVPR2020. The code contains training and testing two binarized object detectors, SSD300 and Faster R-CNN, using our BiDet method on two datasets, PASCAL VOC and Microsoft COCO 2014.

## Update
- 2021.1: Our extended version of BiDet is accepted by T-PAMI! We further improve the performance of binary detectors and extend our method to multi model compression methods. Check it out [here](https://ieeexplore.ieee.org/abstract/document/9319565).
- 2021.4.19: We provide BiDet-SSD300 pretrained weight on Pascal VOC dataset which achieves 66.0% mAP as described in the paper. You can download it [here](https://drive.google.com/file/d/1mURBX-EtoFanp-8wP6-n3u0E6Mq3enkv/view?usp=sharing).

## Quick Start
### Prerequisites
- python 3.6+
- pytorch 1.0+
- other packages include numpy, cv2, matplotlib, pillow, cython, cffi, msgpack, easydict, pyyaml

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

## Frequently Asked Questions

- **What is the difference between BiDet and BiDet (SC)?**

They are two different binary neural networks (BNNs) architectures. As BiDet can be regarded as a training strategy with the IB and sparse object priors loss, we adopt popular BNN methods as our models. BiDet means applying our method to Xnor-Net like architecture, with BN-->BinActiv-->BinConv orders and scaling factors. BiDet (SC) means applying our method to Bi-Real-Net like architecture, with additional shortcuts. **This repo provides implementations of BiDet (SC) for both SSD and Faster R-CNN.**

- **Do you modify the structure of these detection networks?**

For Faster R-CNN with ResNet-18 backbone, we do no modification. **For SSD300 with VGG16 backbone, we restructure it to make it suitable for BNNs.** Please refer to [this issue](https://github.com/ZiweiWangTHU/BiDet/issues/16) for more details.

- **Is the BiDet detectors fully binarized?**

Yes, **both the backbone and detection heads of BiDet detectors are binarized**. One of the main contributions of our work is that we show FULLY binarized object detectors can still get relatively good performance on large-scale datasets such as PASCAL VOC and COCO.

- **How do you calculate the model parameter size and FLOPs?**

I use an open source [PyTorch libary](https://github.com/Lyken17/pytorch-OpCounter) to do so.

- **Why the saved model weight has much larger size than reported in the paper? Why the weight values are not binarized? How about the inference speed?**

Currently there is no official support for binary operations such as Xnor and bitcount in PyTorch, so all BNN researchers use normal weight (float32) to approximate them by binarization at inference time. That is why model size is large and weight values not binarized. As for the inference speed, this is very important for BNNs, but as I said, PyTorch doesn't have acceleration for these operations, so it will be slow using PyTorch. I recommend you to try some BNN inference libraries, such as [daBNN](https://github.com/JDAI-CV/dabnn). Please refer to [this](https://github.com/ZiweiWangTHU/BiDet/issues/13) and [this issue](https://github.com/ZiweiWangTHU/BiDet/issues/1) for more details.

- **The training is not stable.**

Yes. The training of BNNs is known to be unstable and requires fine-tuning. Please refer to [this issue](https://github.com/ZiweiWangTHU/BiDet/issues/8) for more detailed discussions.

## Acknowledgement

We thank the authors of [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) for opening source their wonderful works. We thank [daquexian](https://github.com/daquexian) for providing his implementation of Bi-Real-Net.

## License

BiDet is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Ziyi Wu dazitu616@gmail.com


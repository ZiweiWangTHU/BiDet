# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
import random
import numpy as np
import torch

SEED = 1


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

import os
import sys
import argparse
import pprint
import pdb
import time

from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.faster_rcnn.bidet_resnet import bidet_resnet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a BiDet Faster R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='voc', type=str)
    parser.add_argument('--data_root', dest='data_root',
                        help='path to your dataset',
                        default='none', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        help='backbone for faster-rcnn',
                        default='bidet18', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=50, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)

    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True, type=str2bool)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        default=False, type=str2bool)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=False, type=str2bool)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True, type=str2bool)
    parser.add_argument('--trans_img', dest='trans_img',
                        help='whether transpose image channels from rgb to bgr in order to fit the caffe weight',
                        default=False, type=str2bool)
    parser.add_argument('--fix_num', dest='fix_num',
                        help='number of fixed blocks in backbone',
                        default=0, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=6, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight decay for optimizer',
                        default=0., type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=str2bool)
    parser.add_argument('--basenet', dest='basenet',
                        help='weight path for pretrained backbone network',
                        default="pretrain/resnet18.pth", type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default='none', type=str)

    # IB training setting
    parser.add_argument('--rpn_prior_weight', dest='rpn_prior_weight',
                        help='prior loss weight for -plog(p) in IB',
                        default=0.0, type=float)
    parser.add_argument('--rpn_reg_weight', dest='rpn_reg_weight',
                        help='regularization loss weight on feature map',
                        default=0.0, type=float)
    parser.add_argument('--head_prior_weight', dest='head_prior_weight',
                        help='prior loss weight for -plog(p) in IB',
                        default=0.0, type=float)
    parser.add_argument('--head_reg_weight', dest='head_reg_weight',
                        help='regularization loss weight on feature map',
                        default=0.0, type=float)
    parser.add_argument('--nms_threshold', dest='nms_threshold',
                        help='NMS threshold for IB prior loss -plog(p)',
                        default=0.01, type=float)
    parser.add_argument('--sample_sigma', dest='sample_sigma',
                        help='sigma for sampling loc data',
                        default=0.001, type=float)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "voc":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    else:
        exit(-1)

    args.cfg_file = "faster_rcnn/cfgs/{}_ls.yml".format(
        args.backbone) if args.large_scale else "faster_rcnn/cfgs/{}.yml".format(args.backbone)

    # prepare pretrained backbone path
    basenet_path = args.basenet
    basenet_path = basenet_path if basenet_path.lower() != "none".lower() else None

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.DATA_DIR = args.data_root
    cfg.RESNET.FIXED_BLOCKS = args.fix_num

    print('Using config:')
    pprint.pprint(cfg)

    if args.cuda:
        torch.backends.cudnn.benchmark = True

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    output_dir = "./logs/" + args.dataset + "/" + args.backbone + "_IB/" + str(start_datetime) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes,
                             training=True, trans=args.trans_img)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.backbone == 'bidet18':
        fasterRCNN = bidet_resnet(imdb.classes, 18, class_agnostic=args.class_agnostic,
                                  model_path=basenet_path, nms_threshold=args.nms_threshold,
                                  sample_sigma=args.sample_sigma,
                                  fix_real_conv=True, fix_base_bn=False, fix_top_bn=False,
                                  rpn_prior_weight=args.rpn_prior_weight, rpn_reg_weight=args.rpn_reg_weight,
                                  head_prior_weight=args.head_prior_weight, head_reg_weight=args.head_reg_weight)
    elif args.backbone == 'bidet34':
        fasterRCNN = bidet_resnet(imdb.classes, 34, class_agnostic=args.class_agnostic,
                                  model_path=basenet_path, nms_threshold=args.nms_threshold,
                                  sample_sigma=args.sample_sigma,
                                  fix_real_conv=True, fix_base_bn=False, fix_top_bn=False,
                                  rpn_prior_weight=args.rpn_prior_weight, rpn_reg_weight=args.rpn_reg_weight,
                                  head_prior_weight=args.head_prior_weight, head_reg_weight=args.head_reg_weight)
    else:
        print("network is not defined")
        exit(-1)

    fasterRCNN.create_architecture()

    # prepare for optimizer
    lr = args.lr
    momentum = args.momentum
    cfg.TRAIN.WEIGHT_DECAY = args.weight_decay

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer.lower() == "adam".lower():
        optimizer = torch.optim.Adam(params, lr=args.lr)

    elif args.optimizer.lower() == "sgd".lower():
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=momentum)

    if args.resume:
        print("loading checkpoint %s" % args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        fasterRCNN.load_state_dict(checkpoint['weight'], strict=True)
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        if 'opt' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['opt'])

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = train_size // args.batch_size

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0.
        epoch_loss = 0.
        epoch_rpn_cls_loss = 0.
        epoch_rpn_bbox_loss = 0.
        epoch_rcnn_cls_loss = 0.
        epoch_rcnn_bbox_loss = 0.
        epoch_rpn_prior_loss = 0.
        epoch_rpn_reg_loss = 0.
        epoch_head_prior_loss = 0.
        epoch_head_reg_loss = 0.
        start = time.time()

        if epoch != 1 and (epoch - 1) % args.lr_decay_step == 0 and lr * args.lr_decay_gamma >= 1e-6:
            if lr == args.lr:
                args.rpn_prior_weight = 0.01
                args.rpn_reg_weight = 0.5
                args.head_prior_weight = 0.05
                args.head_reg_weight = 0.02
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_bbox, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
            rpn_prior_loss, rpn_reg_loss, head_prior_loss, head_reg_loss = \
                fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + \
                   rpn_prior_loss.mean() + rpn_reg_loss.mean() + head_prior_loss.mean() + head_reg_loss.mean()

            # inf loss error occurs sometimes on coco dataset
            if loss.item() == float("Inf"):
                print('inf loss error!')
                loss.backward()
                optimizer.zero_grad()
                fasterRCNN.zero_grad()
                torch.cuda.empty_cache()
                continue

            loss_temp += loss.item()
            epoch_loss += loss.item() / float(iters_per_epoch)
            epoch_rpn_cls_loss += (rpn_loss_cls.mean()).item() / float(iters_per_epoch)
            epoch_rpn_bbox_loss += (rpn_loss_bbox.mean()).item() / float(iters_per_epoch)
            epoch_rcnn_cls_loss += (RCNN_loss_cls.mean()).item() / float(iters_per_epoch)
            epoch_rcnn_bbox_loss += (RCNN_loss_bbox.mean()).item() / float(iters_per_epoch)
            rpn_prior_loss = rpn_prior_loss.mean().detach().cpu().item()
            epoch_rpn_prior_loss += rpn_prior_loss / float(iters_per_epoch)
            rpn_reg_loss = rpn_reg_loss.mean().detach().cpu().item()
            epoch_rpn_reg_loss += rpn_reg_loss / float(iters_per_epoch)
            head_prior_loss = head_prior_loss.mean().detach().cpu().item()
            epoch_head_prior_loss += head_prior_loss / float(iters_per_epoch)
            head_reg_loss = head_reg_loss.mean().detach().cpu().item()
            epoch_head_reg_loss += head_reg_loss / float(iters_per_epoch)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_bbox.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_bbox.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, "
                      "\t\t\trpn_prior %.4f, rpn_reg %.4f, head_prior %.4f, head_reg %.4f" %
                      (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,
                       rpn_prior_loss, rpn_reg_loss, head_prior_loss, head_reg_loss))

                loss_temp = 0.
                start = time.time()
                torch.cuda.empty_cache()

        save_name = os.path.join(output_dir, 'model_{}_loss_{}_lr_{}_'
                                             'rpn_cls_{}_rpn_bbox_{}_'
                                             'rcnn_cls_{}_rcnn_bbox_{}_'
                                             'rpn_prior_{}_rpn_reg_{}_'
                                             'head_prior_{}_head_reg_{}.pth'.
                                 format(epoch, round(epoch_loss, 4), lr,
                                        round(epoch_rpn_cls_loss, 4),
                                        round(epoch_rpn_bbox_loss, 4),
                                        round(epoch_rcnn_cls_loss, 4),
                                        round(epoch_rcnn_bbox_loss, 4),
                                        round(epoch_rpn_prior_loss, 4),
                                        round(epoch_rpn_reg_loss, 4),
                                        round(epoch_head_prior_loss, 4),
                                        round(epoch_head_reg_loss, 4)))
        ckp = {
            'weight': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'opt': optimizer.state_dict()
        }
        torch.save(ckp, save_name)
        print('save model: {}'.format(save_name))

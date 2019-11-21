"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import BaseTransformTesting, COCODetectionTesting
from data import COCO_CLASSES as labelmap

from bidet_ssd import build_bidet_ssd

import os
import os.path as osp
import json
import time
import uuid
import pickle
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--weight_path',
                    default='none', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--iou_threshold', default=0.45, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=200, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--coco_root', default="/path/to/coco/",
                    help='Location of COCO root directory')
parser.add_argument('--retest', default=False, type=str2bool,
                    help='test the result on result file')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multi-threading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(save_folder, net, cuda, testset, transform):
    with torch.no_grad():
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # dump predictions and assoc. ground truth to text file for now
        num_images = len(testset)
        num_classes = 81
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        det_file = os.path.join(save_folder, 'detections.pkl')

        if args.retest:
            f = open(det_file, 'rb')
            all_boxes = pickle.load(f)
            print('Evaluating detections')
            testset.evaluate_detections(all_boxes, save_folder)
            return

        for i in range(num_images):
            img, h, w = testset.pull_image(i)
            x = Variable(transform(img).unsqueeze(0))
            if cuda:
                x = x.cuda()

            _t['im_detect'].tic()
            detections = net(x).data  # [1, class, top_k, 5]
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :, :]  # [top_k, 5]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()  # [top_k, 5]
                dets = torch.masked_select(dets, mask).view(-1, 5)  # [top_k, 5]
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time), end='\r')

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_bidet_ssd('test', 300, num_classes, nms_conf_thre=args.confidence_threshold,
                          nms_iou_thre=args.iou_threshold, nms_top_k=args.top_k)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    if args.weight_path.lower() != 'none'.lower():
        print("Loading weight:", args.weight_path)
        net.load_state_dict(torch.load(args.weight_path))
    net.eval()
    print('Finished loading model!')

    # load data
    testset = COCODetectionTesting(args.coco_root, [('2014', 'minival')], None)

    # evaluation
    save_folder = os.path.join(args.save_folder, 'coco')
    test_net(save_folder, net, args.cuda, testset,
             BaseTransformTesting(300, rgb_means=(123, 117, 104), rgb_std=(1, 1, 1), swap=(2, 0, 1)))

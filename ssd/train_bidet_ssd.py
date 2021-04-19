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

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from bidet_ssd import build_bidet_ssd
import os
import time
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

REGULARIZATION_LOSS_WEIGHT = 1.
PRIOR_LOSS_WEIGHT = 1.
NMS_CONF_THRE = 0.03
SQRT_2_PI = math.sqrt(2. * math.pi)
GRADIENT_CLIP_NORM = 5.


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--data_root', default="/path/to/dataset/",
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='./pretrain/vgg16.pth', type=str,
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=False, type=str2bool,
                    help='Whether to resume training from pretrained weights')
parser.add_argument('--weight_path', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0., type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--reg_weight', default=0., type=float,
                    help='regularization loss weight for feature maps')
parser.add_argument('--prior_weight', default=0., type=float,
                    help='loss weight for N(0, 1) prior')
parser.add_argument('--sigma', default=0., type=float,
                    help='scale factor controlling the sample procedure')
parser.add_argument('--nms_conf_threshold', default=0.03, type=float,
                    help='confidence threshold for nms')
parser.add_argument('--opt', default='Adam', type=str,
                    help='Optimizer for training the network')
parser.add_argument('--clip_grad', default=False, type=str2bool,
                    help='whether to clip gradient when training')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logs_dir = os.path.join('logs', args.dataset, str(start_datetime))

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


def train():
    global REGULARIZATION_LOSS_WEIGHT, PRIOR_LOSS_WEIGHT, NMS_CONF_THRE
    if args.dataset == 'COCO':
        cfg = coco
        dataset = COCODetection(root=args.data_root,
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=args.data_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))

    ssd_net = build_bidet_ssd('train', cfg['min_dim'], cfg['num_classes'],
                              nms_conf_thre=NMS_CONF_THRE)
    net = ssd_net

    if args.cuda:
        cudnn.benchmark = True

    opt_state_dict = None
    if args.resume:
        print('Resuming training, loading {}...'.format(args.weight_path))
        try:
            ssd_net.load_state_dict(torch.load(args.weight_path))
        except:  # checkpoint
            print('Extracting from checkpoint')
            ckp = torch.load(args.weight_path, map_location='cpu')
            ssd_net.load_state_dict(ckp['weight'])
            opt_state_dict = ckp['opt']
    else:
        if args.basenet.lower() != 'none':
            vgg_weights = torch.load(args.basenet)
            print('Loading base network...')
            ssd_net.vgg.layers.load_state_dict(vgg_weights, strict=True)

    if args.cuda:
        net = nn.DataParallel(ssd_net).cuda()

    if args.opt.lower() == 'SGD'.lower():
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'Adam'.lower():
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        exit(-1)
    if opt_state_dict is not None:
        print('Load optimizer state dict!')
        optimizer.load_state_dict(opt_state_dict)
        if get_lr(optimizer) != args.lr:
            adjust_learning_rate(optimizer, args.lr)

    optimizer.zero_grad()
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()

    # loss counters
    loss_count = 0.  # for prior loss
    loc_loss_save = 0.
    conf_loss_save = 0.
    reg_loss_save = 0.
    prior_loss_save = 0.
    loss_l = 0.
    loss_c = 0.
    loss_r = 0.
    loss_p = 0.
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        t0 = time.time()

        lr = get_lr(optimizer)

        if iteration % epoch_size == 0 and iteration != 0:
            # reset epoch loss counters
            epoch += 1

        if iteration in cfg['lr_steps']:
            # add our BiDet loss in the after the first lr decay
            if step_index == 0:
                args.reg_weight = 0.1
                args.prior_weight = 0.2
                REGULARIZATION_LOSS_WEIGHT = args.reg_weight
                PRIOR_LOSS_WEIGHT = args.prior_weight
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            print("decay lr")

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            with torch.no_grad():
                images = Variable(images.float().cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            with torch.no_grad():
                images = Variable(images.float())
                targets = [Variable(ann) for ann in targets]

        batch_size = images.size(0)

        if PRIOR_LOSS_WEIGHT != 0.:
            gt_class = [targets[batch_idx][:, -1] for batch_idx in range(batch_size)]

        # forward
        out = net(images)

        loc_data, conf_data, priors, feature_map = out

        # sample loc data from predicted miu and sigma
        normal_dist = torch.randn(loc_data.size(0), loc_data.size(1), 4).float().cuda()
        log_sigma_2 = loc_data[:, :, :4]
        miu = loc_data[:, :, 4:]
        sigma = torch.exp(log_sigma_2 / 2.)
        sample_loc_data = normal_dist * sigma * args.sigma + miu
        loc_data = sample_loc_data

        out = (
            loc_data,
            conf_data,
            priors
        )

        # BP
        loss_l, loss_c = criterion(out, targets)
        loss_temp = loss_l + loss_c

        # COCO dataset bug, maybe due to wrong annotations?
        if loss_temp.item() == float("Inf"):
            print('inf loss error!')
            # the following code is to clear GPU memory for feature_map
            # I don't know other better ways to do so except for BP the loss
            loss_temp.backward()
            net.zero_grad()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        if PRIOR_LOSS_WEIGHT != 0.:
            loss_count = 0.

            detect_result = net.module.detect_prior.forward(
                loc_data,  # localization preds
                net.module.softmax(conf_data),  # confidence preds
                priors,  # default boxes
                gt_class
            )  # [batch, classes, top_k, 5 (score, (y1, x1, y2, x2))]

            num_classes = detect_result.size(1)

            # skip j = 0, because it's the background class
            for j in range(1, num_classes):
                all_dets = detect_result[:, j, :, :]  # [batch, top_k, 5]
                all_mask = all_dets[:, :, :1].gt(0.).expand_as(all_dets)  # [batch, top_k, 5]

                for batch_idx in range(batch_size):
                    # skip non-existed class
                    if not (gt_class[batch_idx] == j - 1).any():
                        continue

                    dets = torch.masked_select(all_dets[batch_idx], all_mask[batch_idx]).view(-1, 5)  # [num, 5]

                    if dets.size(0) == 0:
                        continue

                    # if pred num == gt num, skip
                    if dets.size(0) <= ((gt_class[batch_idx] == j - 1).sum().detach().cpu().item()):
                        continue

                    scores = dets[:, 0]  # [num]
                    scores_sum = scores.sum().item()  # no grad
                    scores = scores / scores_sum  # normalization
                    log_scores = log_func(scores)
                    gt_num = (gt_class[batch_idx] == j - 1).sum().detach().cpu().item()
                    loss_p += (-1. * log_scores.sum() / float(gt_num))
                    loss_count += 1.

            loss_p /= (loss_count + 1e-6)
            loss_p *= PRIOR_LOSS_WEIGHT

        # Calculate regularization loss on feature maps
        # directly use L2 loss here
        if REGULARIZATION_LOSS_WEIGHT != 0.:
            f_num = len(feature_map)
            loss_r = 0.

            for f_m in feature_map:
                loss_r += (f_m ** 2).mean()

            loss_r *= REGULARIZATION_LOSS_WEIGHT
            loss_r /= float(f_num)

        loss = loss_l + loss_c + loss_r + loss_p

        # COCO dataset bug, maybe due to wrong annotations?
        if loss.item() == float("Inf"):
            print('inf loss error!')
            # the following code is to clear GPU memory for feature_map
            # I don't know other better ways to do so except for BP the loss
            loss.backward()
            net.zero_grad()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        # compute gradient and do optimizer step
        loss.backward()
        # clip gradient because binary net training is very unstable
        if args.clip_grad:
            grad_norm = get_grad_norm(net)
            nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_l = loss_l.detach().cpu().item()
        loss_c = loss_c.detach().cpu().item()
        if REGULARIZATION_LOSS_WEIGHT != 0.:
            loss_r = loss_r.detach().cpu().item()
        if PRIOR_LOSS_WEIGHT != 0.:
            loss_p = loss_p.detach().cpu().item()
        loc_loss_save += loss_l
        conf_loss_save += loss_c
        reg_loss_save += loss_r
        prior_loss_save += loss_p
        t1 = time.time()

        if iteration % 100 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter:', iteration, 'loss:', round(loss.detach().cpu().item(), 4))
            print('conf_loss:', round(loss_c, 4), 'loc_loss:', round(loss_l, 4),
                  'reg_loss:', round(loss_r, 4), 'prior_loss:', round(loss_p, 4),
                  'lr:', lr)
            if args.clip_grad:
                print('gradient norm:', grad_norm)
            torch.cuda.empty_cache()

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)

            loss_save = loc_loss_save + conf_loss_save + reg_loss_save + prior_loss_save
            checkpoint = {
                'weight': net.module.state_dict(),
                'opt': optimizer.state_dict()
            }
            torch.save(checkpoint, logs_dir + '/model_' + str(iteration) +
                       '_loc_' + str(round(loc_loss_save / 5000., 4)) +
                       '_conf_' + str(round(conf_loss_save / 5000., 4)) +
                       '_reg_' + str(round(reg_loss_save / 5000., 4)) +
                       '_prior_' + str(round(prior_loss_save / 5000., 4)) +
                       '_loss_' + str(round(loss_save / 5000., 4)) +
                       '_lr_' + str(round(args.lr * (args.gamma ** step_index), 6)) + '.pth')

            loc_loss_save = 0.
            conf_loss_save = 0.
            reg_loss_save = 0.
            prior_loss_save = 0.

        loss_l = 0.
        loss_c = 0.
        loss_r = 0.
        loss_p = 0.
        loss_count = 0.

    torch.save(net.module.state_dict(), logs_dir + '/' + args.dataset + '_final.pth')


def log_func(tensor):
    return tensor * torch.log(tensor)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate of optimizer to new_lr."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_grad_norm(model):
    """Show the max gradient in a step of all the model's parameters."""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            module_norm = p.grad.norm()
            total_norm += module_norm ** 2
    total_norm = torch.sqrt(total_norm).item()
    return total_norm


if __name__ == '__main__':
    REGULARIZATION_LOSS_WEIGHT = args.reg_weight
    PRIOR_LOSS_WEIGHT = args.prior_weight
    NMS_CONF_THRE = args.nms_conf_threshold

    train()

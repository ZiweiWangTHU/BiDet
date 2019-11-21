import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN, _RPN_BiDet
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.roi_layers import nms

from lib.model.roi_layers import ROIAlign, ROIPool

from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import pdb
from lib.model.utils.net_utils import _smooth_l1_loss


class _fasterRCNN_BiDet(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, sample_sigma=0.001,
                 nms_threshold=0.01, rpn_prior_weight=0.2, rpn_reg_weight=0.1,
                 head_prior_weight=0.2, head_reg_weight=0.1):

        super(_fasterRCNN_BiDet, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.sample_sigma = sample_sigma
        # loss
        self.RCNN_loss_cls = torch.FloatTensor([0.]).cuda()
        self.RCNN_loss_bbox = torch.FloatTensor([0.]).cuda()

        # IB setting
        self.nms_threshold = nms_threshold
        self.rpn_prior_weight = rpn_prior_weight
        self.head_prior_weight = head_prior_weight
        self.head_reg_weight = head_reg_weight

        # define rpn
        self.RCNN_rpn = _RPN_BiDet(self.dout_base_model, reg_weight=rpn_reg_weight)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox, fg_scores, rpn_reg_loss = \
            self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        rpn_prior_loss = torch.FloatTensor([0.]).cuda()

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            if self.rpn_prior_weight != 0.:
                for i in range(batch_size):
                    gt_num = num_boxes[i].detach().cpu().item()
                    score = fg_scores[i]
                    score_sum = score.sum().detach().cpu().item()
                    score = score / score_sum
                    log_score = score * torch.log(score + 1e-6)  # p * log(p)
                    rpn_prior_loss += (-1. * log_score.sum() / float(gt_num))

                rpn_prior_loss /= batch_size
                rpn_prior_loss *= self.rpn_prior_weight
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = torch.FloatTensor([0.]).cuda()
            rpn_loss_bbox = torch.FloatTensor([0.]).cuda()

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        head_reg_loss = torch.FloatTensor([0.]).cuda()
        if self.training and self.head_reg_weight != 0.:
            head_reg_loss = (pooled_feat ** 2).mean() * self.head_reg_weight

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        # sample loc data
        normal_dist = torch.randn(bbox_pred.size(0), 4).float().cuda()
        log_sigma_2 = bbox_pred[:, :4]
        miu = bbox_pred[:, 4:]
        sigma = torch.exp(log_sigma_2 / 2.)
        sample_loc_data = normal_dist * sigma * self.sample_sigma + miu
        bbox_pred = sample_loc_data

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = torch.FloatTensor([0.]).cuda()
        RCNN_loss_bbox = torch.FloatTensor([0.]).cuda()

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        head_prior_loss = torch.FloatTensor([0.]).cuda()
        if self.training and self.head_prior_weight != 0.:
            scores = cls_prob.data  # [batch, num_rois, classes]
            scores_gradient = cls_prob  # [batch, num_rois, classes]
            boxes = rois.data[:, :, 1:5]  # [batch, num_rois, 4]
            if cfg.TRAIN.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data  # [batch, num_rois, 4]
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(batch_size, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(batch_size, -1, 4 * len(self.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, batch_size)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, batch_size)
            else:
                # Simply repeat the boxes, once for each class
                print("no use bbox head in IB")
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_info[:, 2].data[:, None, None]  # [batch, num_rois, 4]
            loss_count = 0.
            gt_classes = gt_boxes[:, :, -1].data  # [batch, num(0 pad to 20)]
            for i in range(batch_size):
                for j in range(1, len(self.classes)):  # skip background class
                    if not (gt_classes[i] == j).any():  # no such class in gt
                        continue
                    # there are gt for this class
                    inds = torch.nonzero(scores[i, :, j] > self.nms_threshold).view(-1)
                    if inds.numel() == 0:
                        continue
                    cls_scores = scores[i, :, j][inds]  # [num]
                    cls_scores_gradient = scores_gradient[i, :, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.class_agnostic:
                        cls_boxes = pred_boxes[i, inds, :]  # [num, 4]
                    else:
                        cls_boxes = pred_boxes[i, inds][:, j * 4:(j + 1) * 4]
                    cls_scores_gradient = cls_scores_gradient[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    score = cls_scores_gradient[keep.view(-1).long()]  # [num_keep]
                    gt_num = (gt_classes[i] == j).sum().detach().cpu().item()
                    if score.size(0) <= gt_num:
                        continue
                    score_sum = score.sum().detach().cpu().item()
                    score = score / score_sum
                    log_score = score * torch.log(score + 1e-6)
                    head_prior_loss += (-1. * log_score.sum() / float(gt_num))
                    loss_count += 1.

            head_prior_loss /= loss_count
            head_prior_loss *= self.head_prior_weight

        return rois, cls_prob, bbox_pred, \
               rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               rpn_prior_loss, rpn_reg_loss, head_prior_loss, head_reg_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

import torch
from ..box_utils import decode, nms
from data import voc as cfg


class Detect:
    """At test time, Detect is the final layer of   Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k=200, conf_thresh=0.03, nms_thresh=0.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def __call__(self, loc_data, conf_data, prior_data):
        return self.forward(loc_data, conf_data, prior_data)

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
            gt_class: [batch_size, num_gt], class labels for every image, we only do nms on gt classes!
        """
        prior_data = prior_data[:loc_data.size(1)]
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(batch_size, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output


class DetectPrior:
    """Almost the same as class Detect, except that we only process gt-classes.
    Use this class when calculating prior loss.
    """

    def __init__(self, num_classes, bkg_label, top_k=200, conf_thresh=0.03, nms_thresh=0.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def __call__(self, loc_data, conf_data, prior_data, gt_class=None):
        return self.forward(loc_data, conf_data, prior_data, gt_class)

    def forward(self, loc_data, conf_data, prior_data, gt_class=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
            gt_class: [batch_size, num_gt], class labels for every image, we only do nms on gt classes!
        """
        prior_data = prior_data[:loc_data.size(1)]
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(batch_size, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # perform nms only for gt class when calculating prior loss
                if gt_class is not None:
                    if not (gt_class[i] == cl - 1).any():
                        continue

                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes.data, scores.data, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output

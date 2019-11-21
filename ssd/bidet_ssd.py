import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import *
from data import voc, coco
from binary_utils import BinarizeConv2d

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
EXTRAS = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
MBOX = [4, 6, 6, 6, 4, 4]


class BiDetVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 downsample=None):
        super(BiDetVGGBlock, self).__init__()
        self.conv = BinarizeConv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.bn(self.conv(x))

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual

        return x


def bidet_vgg(cfg, i=3, bias=False):
    layers = []
    in_channels = i
    for j in range(len(cfg)):
        v = cfg[j]
        if v == 'M':
            layers += [BiDetVGGBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                     downsample=nn.Sequential(
                                         nn.AvgPool2d(kernel_size=2, stride=2),
                                         nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                                   kernel_size=1, stride=1, padding=0, bias=bias),
                                         nn.BatchNorm2d(cfg[j + 1])
                                     ))]
            in_channels = cfg[j + 1]
        elif v == 'C':
            layers += [BiDetVGGBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                     downsample=nn.Sequential(
                                         nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                         nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                                   kernel_size=1, stride=1, padding=0, bias=bias),
                                         nn.BatchNorm2d(cfg[j + 1])
                                     ))]
            in_channels = cfg[j + 1]
        else:
            if in_channels == i:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=bias)
                layers += [conv, nn.BatchNorm2d(v)]
            else:
                layers += [BiDetVGGBlock(in_channels, v, kernel_size=3, padding=1, bias=bias)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = BinarizeConv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # shouldn't binarize fc layer
    layers += [pool5]
    layers += [conv6, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    layers += [conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]

    return layers


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, BinarizeConv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


class VGGBase(nn.Module):

    def __init__(self, base):
        super(VGGBase, self).__init__()
        self.layers = nn.ModuleList(base)

    def forward(self, x):
        for i in range(14):
            x = self.layers[i](x)
        output1 = x
        for i in range(14, len(self.layers)):
            x = self.layers[i](x)
        output2 = x

        return output1, output2


class BiDetSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,
                 nms_conf_thre=0.03, nms_iou_thre=0.45, nms_top_k=200):
        super(BiDetSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = VGGBase(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extra_bns = nn.ModuleList([nn.BatchNorm2d(self.extras[i].out_channels)
                                        for i in range(len(self.extras))])

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect_prior = DetectPrior(num_classes, 0, nms_top_k, nms_conf_thre, nms_iou_thre)

        if self.phase == "test":
            self.detect = Detect(num_classes, 0, nms_top_k, nms_conf_thre, nms_iou_thre)

        init_model(self)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        output1, x = self.vgg(x)

        s = self.L2Norm(output1)
        sources.append(s)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for i in range(len(self.extras)):
            x = self.extra_bns[i](self.extras[i](x))
            if i % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                (loc.view(loc.size(0), -1, 8)[:, :, 4:]),  # loc data without sigma, directly use mean
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 8),
                conf.view(conf.size(0), -1, self.num_classes),  # [batch, Σ(H * W * N), num_classes]
                self.priors,
                sources
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [BinarizeConv2d(in_channels, cfg[k + 1],
                                          kernel_size=(1, 3)[flag], stride=2, padding=1,
                                          bias=False)]
            else:
                layers += [BinarizeConv2d(in_channels, v, kernel_size=(1, 3)[flag],
                                          bias=False)]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    """This function constructs the detector heads of ssd, so may not be quantized"""
    loc_layers = []
    conf_layers = []
    vgg_source = [13, -3]
    for k, v in enumerate(vgg_source):
        try:
            channel = vgg[v].out_channels
        except:
            channel = vgg[v].conv.out_channels

        loc_layers += [nn.Conv2d(channel, cfg[k] * 8, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channel, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 8, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_bidet_ssd(phase, size=300, num_classes=21,
                    nms_conf_thre=0.01, nms_iou_thre=0.45, nms_top_k=200):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(bidet_vgg(CFG, 3, bias=False),
                                     add_extras(EXTRAS, 1024),
                                     MBOX, num_classes)
    return BiDetSSD(phase, size, base_, extras_, head_, num_classes,
                    nms_conf_thre, nms_iou_thre, nms_top_k)

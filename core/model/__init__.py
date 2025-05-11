import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import  ResNet, Bottleneck, BasicBlock
from .wideresnet import WideResNet

def build_model_res50gn(group_norm, num_classes):
    print('Building model...')
    def gn_helper(planes):
        return nn.BatchNorm2d(planes)
        # return nn.GroupNorm(group_norm, planes)
    net = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, norm_layer=gn_helper)
    return net

def convert_batchnorm_to_groupnorm(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
            setattr(model, name, gn)
        else:
            convert_batchnorm_to_groupnorm(module, num_groups)
    return model

def build_model_res18bn(num_classes):
    print('Building model...')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=nn.BatchNorm2d)

def build_model_wrn2810bn(num_classes):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import  ResNet, Bottleneck, BasicBlock
from .custom_resnet import ResNet as ResNetTin, BasicBlock as BasicBlockTin

def build_model_res50gn(group_norm, num_classes):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(group_norm, planes)
    net = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, norm_layer=gn_helper)
    return net

def build_model_res18bn(num_classes):
    print('Building model...')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=nn.BatchNorm2d)

def build_model_wrn2810bn(num_classes):
    pass

def build_model_resnet18_TIN(num_classes):
    print("Using custom ResNet18 for TinyImageNet")
    return ResNetTin('tiny', BasicBlockTin, [2, 2, 2, 2])  # or however it's defined in your model
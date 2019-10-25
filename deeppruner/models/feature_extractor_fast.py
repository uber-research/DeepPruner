# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodules import BasicBlock, convbn_relu


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn_relu(3, 32, 3, 2, 1, 1),
                                       convbn_relu(32, 32, 3, 1, 1, 1),
                                       convbn_relu(32, 32, 3, 1, 1, 1))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn_relu(128, 32, 1, 1, 0, 1))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn_relu(128, 32, 1, 1, 0, 1))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn_relu(128, 32, 1, 1, 0, 1))

        self.lastconv = nn.Sequential(convbn_relu(352, 128, 3, 1, 1, 1),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        """
        Feature Extractor
        Description:    The goal of the feature extraction network is to produce a reliable pixel-wise
                        feature representation from the input image. Specifically, we employ four residual blocks
                        and use X2 dilated convolution for the last block to enlarge the receptive field.
                        We then apply spatial pyramid pooling to build a 4-level pyramid feature.
                        Through multi-scale information, the model is able to capture large context while
                        maintaining a high spatial resolution. The size of the final feature map is 1/4 of
                        the originalinput image size. We share the parameters for the left and right feature network.

        Args:
            :input: Input image (RGB)

        Returns:
            :output_feature: spp_features (downsampled X8)
            :output_raw: features (downsampled X4)
            :output1: low_level_features (downsampled X2)
        """

        output0 = self.firstconv(input)
        output1 = self.layer1(output0)
        output_raw = self.layer2(output1)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat((output, output_skip, output_branch4, output_branch3, output_branch2), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature, output_raw, output1

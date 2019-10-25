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
import torch.nn as nn
from models.submodules import SubModule, convbn_2d_lrelu


class RefinementNet(SubModule):
    def __init__(self, inplanes):
        super(RefinementNet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d_lrelu(inplanes, 32, kernel_size=3, stride=1, pad=1),
            convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            convbn_2d_lrelu(32, 16, kernel_size=3, stride=1, pad=2, dilation=2),
            convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=4, dilation=4),
            convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=1, dilation=1))

        self.classif1 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.weight_init()

    def forward(self, input, disparity):
        """
        Refinement Block
        Description:    The network takes left image convolutional features from the second residual block
                        of the feature network and the current disparity estimation as input.
                        It then outputs the finetuned disparity prediction. The low-level feature
                        information serves as a guidance to reduce noise and improve the quality of the final
                        disparity map, especially on sharp boundaries.

        Args:
            :input: Input features composed of left image low-level features, cost-aggregator features, and
                    cost-aggregator disparity.

            :disparity: predicted disparity
        """

        output0 = self.conv1(input)
        output0 = self.classif1(output0)
        output = self.relu(output0 + disparity)

        return output
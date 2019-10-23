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
import torch.nn.functional as F


class feature_extractor(nn.Module):
    def __init__(self, filter_size):
        super(feature_extractor, self).__init__()

        self.filter_size = filter_size

    def forward(self, left_input, right_input):
        """
        Feature Extractor

        Description: Aggregates the RGB values from the neighbouring pixels in the window (filter_size * filter_size).
                    No weights are learnt for this feature extractor.

        Args:
            :param left_input: Left Image
            :param right_input: Right Image

        Returns:
            :left_features: Left Image features
            :right_features: Right Image features
            :one_hot_filter: Convolution filter used to aggregate neighbour RGB features to the center pixel.
                             one_hot_filter.shape = (filter_size * filter_size)
        """

        device = left_input.get_device()

        label = torch.arange(0, self.filter_size * self.filter_size, device=device).repeat(
            self.filter_size * self.filter_size).view(
            self.filter_size * self.filter_size, 1, 1, self.filter_size, self.filter_size)

        one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()

        left_features = F.conv3d(left_input.unsqueeze(1), one_hot_filter,
                                 padding=(0, self.filter_size // 2, self.filter_size // 2))
        right_features = F.conv3d(right_input.unsqueeze(1), one_hot_filter,
                                  padding=(0, self.filter_size // 2, self.filter_size // 2))

        left_features = left_features.view(left_features.size()[0],
                                           left_features.size()[1] * left_features.size()[2],
                                           left_features.size()[3],
                                           left_features.size()[4])

        right_features = right_features.view(right_features.size()[0],
                                             right_features.size()[1] * right_features.size()[2],
                                             right_features.size()[3],
                                             right_features.size()[4])

        return left_features, right_features, one_hot_filter

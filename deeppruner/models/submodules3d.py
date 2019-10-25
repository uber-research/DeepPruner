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
from models.submodules import SubModule, convbn_3d_lrelu, convbn_transpose_3d


class HourGlass(SubModule):
    def __init__(self, inplanes=16):
        super(HourGlass, self).__init__()

        self.conv1 = convbn_3d_lrelu(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = convbn_3d_lrelu(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv1_1 = convbn_3d_lrelu(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = convbn_3d_lrelu(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1)

        self.conv3 = convbn_3d_lrelu(inplanes * 4, inplanes * 8, kernel_size=3, stride=2, pad=1)
        self.conv4 = convbn_3d_lrelu(inplanes * 8, inplanes * 8, kernel_size=3, stride=1, pad=1)

        self.conv5 = convbn_transpose_3d(inplanes * 8, inplanes * 4, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv6 = convbn_transpose_3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv7 = convbn_transpose_3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                         output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False)

        self.last_conv3d_layer = nn.Sequential(
            convbn_3d_lrelu(inplanes, inplanes * 2, 3, 1, 1),
            nn.Conv3d(inplanes * 2, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.softmax = nn.Softmax(dim=1)

        self.weight_init()


class MaxDisparityPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(MaxDisparityPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input, input_disparity):
        """
        Confidence Range Prediction (Max Disparity):
        Description:    The network has a convolutional encoder-decoder structure. It takes the sparse
                disparity estimations from the differentiable PatchMatch, the left image and the warped right image
                (warped according to the sparse disparity estimations) as input and outputs the upper bound of
                the confidence range for each pixel i.
        Args:
            :input: Left and Warped right Image features as Cost Volume.
            :input_disparity: PatchMatch predicted disparity samples.
        Returns:
            :disparity_output: Max Disparity of the reduced disaprity search range.
            :feature_output:   High-level features of the MaxDisparityPredictor
        """

        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0

        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0

        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0

        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1)

        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2

        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1).unsqueeze(1)

        return disparity_output, feature_output


class MinDisparityPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(MinDisparityPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input, input_disparity):
        """
        Confidence Range Prediction (Min Disparity):
        Description:    The network has a convolutional encoder-decoder structure. It takes the sparse
                disparity estimations from the differentiable PatchMatch, the left image and the warped right image
                (warped according to the sparse disparity estimations) as input and outputs the lower bound of
                the confidence range for each pixel i.
        Args:
            :input: Left and Warped right Image features as Cost Volume.
            :input_disparity: PatchMatch predicted disparity samples.
        Returns:
            :disparity_output: Min Disparity of the reduced disaprity search range.
            :feature_output:   High-level features of the MaxDisparityPredictor
        """

        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0

        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0

        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0

        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1)

        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2

        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1).unsqueeze(1)

        return disparity_output, feature_output


class CostAggregator(HourGlass):

    def __init__(self, cost_aggregator_inplanes, hourglass_inplanes=16):
        super(CostAggregator, self).__init__(inplanes=16)

        self.dres0 = nn.Sequential(convbn_3d_lrelu(cost_aggregator_inplanes, 64, 3, 1, 1),
                                   convbn_3d_lrelu(64, 32, 3, 1, 1))

        self.dres1 = nn.Sequential(convbn_3d_lrelu(32, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, hourglass_inplanes, 3, 1, 1))

    def forward(self, input, input_disparity):
        """
        3D Cost Aggregator
        Description:    Based on the predicted range in the pruning module,
                we build the 3D cost volume estimator and conduct spatial aggregation.
                Following common practice, we take the left image, the warped right image and corresponding disparities
                as input and output the cost over the disparity range at the size B X R X H X W , where R is the number
                of disparities per pixel. Compared to prior work, our R is more than 10 times smaller, making
                this module very efficient. Soft-arg max is again used to predict the disparity value ,
                so that our approach is end-to-end trainable.

        Args:
            :input:   Cost-Volume composed of left image features, warped right image features,
                      Confidence range Predictor features and input disparity samples/

            :input_disparity: input disparity samples.

        Returns:
            :disparity_output: Predicted disparity
            :feature_output: High-level features of 3d-Cost Aggregator

        """

        output0 = self.dres0(input)
        output0_b = self.dres1(output0)

        output0 = self.conv1(output0_b)
        output0_a = self.conv2(output0) + output0

        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0

        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0

        output1 = self.conv5(output0) + output0_c
        output1 = self.conv6(output1) + output0_a
        output1 = self.conv7(output1) + output0_b

        output2 = self.last_conv3d_layer(output1).squeeze(1)
        feature_output = output2

        confidence_output = self.softmax(output2)
        disparity_output = torch.sum(confidence_output * input_disparity, dim=1)

        return disparity_output.unsqueeze(1), feature_output

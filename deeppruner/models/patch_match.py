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


class DisparityInitialization(nn.Module):

    def __init__(self):
        super(DisparityInitialization, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_intervals=10):
        """
        PatchMatch Initialization Block
        Description:    Rather than allowing each sample/ particle to reside in the full disparity space,
                        we divide the search space into 'number_of_intervals' intervals, and force the
                        i-th particle to be in a i-th interval. This guarantees the diversity of the
                        particles and helps improve accuracy for later computations.

                        As per implementation,
                        this function divides the complete disparity search space into multiple intervals.

        Args:
            :min_disparity: Min Disparity of the disparity search range.
            :max_disparity: Max Disparity of the disparity search range.
            :number_of_intervals (default: 10): Number of samples to be generated.
        Returns:
            :interval_noise: Random value between 0-1. Represents offset of the from the interval_min_disparity.
            :interval_min_disparity: disparity_sample = interval_min_disparity + interval_noise
            :multiplier: 1.0 / number_of_intervals
        """

        device = min_disparity.get_device()

        multiplier = 1.0 / number_of_intervals
        range_multiplier = torch.arange(0.0, 1, multiplier, device=device).view(number_of_intervals, 1, 1)
        range_multiplier = range_multiplier.repeat(1, min_disparity.size()[2], min_disparity.size()[3])

        interval_noise = min_disparity.new_empty(min_disparity.size()[0], number_of_intervals, min_disparity.size()[2],
                                                 min_disparity.size()[3]).uniform_(0, 1)
        interval_min_disparity = min_disparity + (max_disparity - min_disparity) * range_multiplier

        return interval_noise, interval_min_disparity, multiplier


class Evaluate(nn.Module):
    def __init__(self, filter_size=3, temperature=7):
        super(Evaluate, self).__init__()
        self.temperature = temperature
        self.filter_size = filter_size
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, left_input, right_input, disparity_samples, normalized_disparity_samples):
        """
        PatchMatch Evaluation Block
        Description:    For each pixel i, matching scores are computed by taking the inner product between the
            left feature and the right feature: score(i,j) = feature_left(i), feature_right(i+disparity(i,j))
            for all candidates j. The best k disparity value for each pixel is carried towards the next iteration.

            As per implementation,
            the complete disparity search range is discretized into intervals in
            DisparityInitialization() function. Corresponding to each disparity interval, we have multiple samples
            to evaluate. The best disparity sample per interval is the output of the function.

        Args:
            :left_input: Left Image Feature Map
            :right_input: Right Image Feature Map
            :disparity_samples: Disparity Samples to be evaluated. For each pixel, we have
                                ("number of intervals" X "number_of_samples_per_intervals") samples.

            :normalized_disparity_samples:
        Returns:
            :disparity_samples: Evaluated disparity sample, one per disparity interval.
            :normalized_disparity_samples: Evaluated normaized disparity sample, one per disparity interval.
        """
        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(
            left_input.size()[2]).view(left_input.size()[2], left_input.size()[3])

        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_sample_strength = disparity_samples.new(disparity_samples.size()[0],
                                                          disparity_samples.size()[1],
                                                          disparity_samples.size()[2],
                                                          disparity_samples.size()[3])

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]).float()
        right_y_coordinate = right_y_coordinate - disparity_samples
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)

        warped_right_feature_map = torch.gather(right_feature_map,
                                                dim=4,
                                                index=right_y_coordinate.expand(
                                                    right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())

        disparity_sample_strength = torch.mean(left_feature_map * warped_right_feature_map, dim=1) * self.temperature

        disparity_sample_strength = disparity_sample_strength.view(
            disparity_sample_strength.size()[0],
            disparity_sample_strength.size()[1] // (self.filter_size),
            (self.filter_size),
            disparity_sample_strength.size()[2],
            disparity_sample_strength.size()[3])

        disparity_samples = disparity_samples.view(disparity_samples.size()[0],
                                                   disparity_samples.size()[1] // (self.filter_size),
                                                   (self.filter_size),
                                                   disparity_samples.size()[2],
                                                   disparity_samples.size()[3])

        normalized_disparity_samples = normalized_disparity_samples.view(
            normalized_disparity_samples.size()[0],
            normalized_disparity_samples.size()[1] // (self.filter_size),
            (self.filter_size),
            normalized_disparity_samples.size()[2],
            normalized_disparity_samples.size()[3])

        disparity_sample_strength = disparity_sample_strength.permute([0, 2, 1, 3, 4])
        disparity_samples = disparity_samples.permute([0, 2, 1, 3, 4])
        normalized_disparity_samples = normalized_disparity_samples.permute([0, 2, 1, 3, 4])

        disparity_sample_strength = torch.softmax(disparity_sample_strength, dim=1)
        disparity_samples = torch.sum(disparity_samples * disparity_sample_strength, dim=1)
        normalized_disparity_samples = torch.sum(normalized_disparity_samples * disparity_sample_strength, dim=1)

        return normalized_disparity_samples, disparity_samples


class Propagation(nn.Module):
    def __init__(self, filter_size=3):
        super(Propagation, self).__init__()
        self.filter_size = filter_size

    def forward(self, disparity_samples, device, propagation_type="horizontal"):
        """
        PatchMatch Propagation Block
        Description:    Particles from adjacent pixels are propagated together through convolution with a
            pre-defined one-hot filter pattern, which en-codes the fact that we allow each pixel
            to propagate particles to its 4-neighbours.

            As per implementation, the complete disparity search range is discretized into intervals in
            DisparityInitialization() function.
            Now, propagation of samples from neighbouring pixels, is done per interval. This implies that after
            propagation, number of samples per pixel = (filter_size X number_of_intervals)

        Args:
            :disparity_samples:
            :device: Cuda device
            :propagation_type (default:"horizontal"): In order to be memory efficient, we use separable convolutions
                                                    for propagtaion.

        Returns:
            :aggregated_disparity_samples: Disparity Samples aggregated from the neighbours.

        """

        disparity_samples = disparity_samples.view(disparity_samples.size()[0],
                                                   1,
                                                   disparity_samples.size()[1],
                                                   disparity_samples.size()[2],
                                                   disparity_samples.size()[3])

        if propagation_type is "horizontal":
            label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(
                self.filter_size, 1, 1, 1, self.filter_size)

            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            aggregated_disparity_samples = F.conv3d(disparity_samples,
                                                    one_hot_filter, padding=(0, 0, self.filter_size // 2))

        else:
            label = torch.arange(0, self.filter_size, device=device).repeat(self.filter_size).view(
                self.filter_size, 1, 1, self.filter_size, 1).long()

            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            aggregated_disparity_samples = F.conv3d(disparity_samples,
                                                    one_hot_filter, padding=(0, self.filter_size // 2, 0))

        aggregated_disparity_samples = aggregated_disparity_samples.permute([0, 2, 1, 3, 4])
        aggregated_disparity_samples = aggregated_disparity_samples.contiguous().view(
            aggregated_disparity_samples.size()[0],
            aggregated_disparity_samples.size()[1] * aggregated_disparity_samples.size()[2],
            aggregated_disparity_samples.size()[3],
            aggregated_disparity_samples.size()[4])

        return aggregated_disparity_samples


class PatchMatch(nn.Module):
    def __init__(self, propagation_filter_size=3):
        super(PatchMatch, self).__init__()

        self.propagation_filter_size = propagation_filter_size
        self.propagation = Propagation(filter_size=propagation_filter_size)
        self.disparity_initialization = DisparityInitialization()
        self.evaluate = Evaluate(filter_size=propagation_filter_size)

    def forward(self, left_input, right_input, min_disparity, max_disparity, sample_count=10, iteration_count=3):
        """
        Differntail PatchMatch Block
        Description:    In this work, we unroll generalized PatchMatch as a recurrent neural network,
                        where each unrolling step is equivalent to each iteration of the algorithm.
                        This is important as it allow us to train our full model end-to-end.
                        Specifically, we design the following layers:
                            - Initialization or Paticle Sampling
                            - Propagation
                            - Evaluation
        Args:
            :left_input: Left Image feature map
            :right_input: Right image feature map
            :min_disparity: Min of the disparity search range
            :max_disparity: Max of the disparity search range
            :sample_count (default:10): Number of disparity samples per pixel. (similar to generalized PatchMatch)
            :iteration_count (default:3) : Number of PatchMatch iterations

        Returns:
            :disparity_samples: For each pixel, this function returns "sample_count" disparity samples.
        """

        device = left_input.get_device()
        min_disparity = torch.floor(min_disparity)
        max_disparity = torch.ceil(max_disparity)

        # normalized_disparity_samples: Disparity samples normalized by the corresponding interval size.
        #                               i.e (disparity_sample - interval_min_disparity) / interval_size

        normalized_disparity_samples, min_disp_tensor, multiplier = self.disparity_initialization(
            min_disparity, max_disparity, sample_count)
        min_disp_tensor = min_disp_tensor.unsqueeze(2).repeat(1, 1, self.propagation_filter_size, 1, 1).view(
            min_disp_tensor.size()[0],
            min_disp_tensor.size()[1] * self.propagation_filter_size,
            min_disp_tensor.size()[2],
            min_disp_tensor.size()[3])

        for prop_iter in range(iteration_count):
            normalized_disparity_samples = self.propagation(normalized_disparity_samples, device, propagation_type="horizontal")
            disparity_samples = normalized_disparity_samples * \
                (max_disparity - min_disparity) * multiplier + min_disp_tensor

            normalized_disparity_samples, disparity_samples = self.evaluate(left_input,
                                                                            right_input,
                                                                            disparity_samples,
                                                                            normalized_disparity_samples)

            normalized_disparity_samples = self.propagation(normalized_disparity_samples, device, propagation_type="vertical")
            disparity_samples = normalized_disparity_samples * \
                (max_disparity - min_disparity) * multiplier + min_disp_tensor

            normalized_disparity_samples, disparity_samples = self.evaluate(left_input,
                                                                            right_input,
                                                                            disparity_samples,
                                                                            normalized_disparity_samples)

        return disparity_samples

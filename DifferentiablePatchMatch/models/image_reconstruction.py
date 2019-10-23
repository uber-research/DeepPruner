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
from models.patch_match import PatchMatch
from models.feature_extractor import feature_extractor
from models.config import config as args


class Reconstruct(nn.Module):
    def __init__(self, filter_size):
        super(Reconstruct, self).__init__()
        self.filter_size = filter_size

    def forward(self, right_input, offset_x, offset_y, x_coordinate, y_coordinate, neighbour_extraction_filter):
        """
        Reconstruct the left image using the NNF(NNF represented by the offsets and the xy_coordinates)
        We did Patch Voting on the offset field, before reconstruction, in order to
                generate smooth reconstruction.
        Args:
            :right_input: Right Image
            :offset_x: horizontal offset to generate the NNF.
            :offset_y: vertical offset to generate the NNF.
            :x_coordinate: X coordinate
            :y_coordinate: Y coordinate

        Returns:
            :reconstruction: Right image reconstruction
        """

        pad_size = self.filter_size // 2
        smooth_offset_x = nn.ReflectionPad2d(
            (pad_size, pad_size, pad_size, pad_size))(offset_x)
        smooth_offset_y = nn.ReflectionPad2d(
            (pad_size, pad_size, pad_size, pad_size))(offset_y)

        smooth_offset_x = F.conv2d(smooth_offset_x,
                                   neighbour_extraction_filter,
                                   padding=(pad_size, pad_size))[:, :, pad_size:-pad_size, pad_size:-pad_size]

        smooth_offset_y = F.conv2d(smooth_offset_y,
                                   neighbour_extraction_filter,
                                   padding=(pad_size, pad_size))[:, :, pad_size:-pad_size, pad_size:-pad_size]

        coord_x = torch.clamp(
            x_coordinate - smooth_offset_x,
            min=0,
            max=smooth_offset_x.size()[3] - 1)

        coord_y = torch.clamp(
            y_coordinate - smooth_offset_y,
            min=0,
            max=smooth_offset_x.size()[2] - 1)

        coord_x -= coord_x.size()[3] / 2
        coord_x /= (coord_x.size()[3] / 2)

        coord_y -= coord_y.size()[2] / 2
        coord_y /= (coord_y.size()[2] / 2)

        grid = torch.cat((coord_x.unsqueeze(4), coord_y.unsqueeze(4)), dim=4)
        grid = grid.view(grid.size()[0] * grid.size()[1], grid.size()[2], grid.size()[3], grid.size()[4])
        reconstruction = F.grid_sample(right_input.repeat(grid.size()[0], 1, 1, 1), grid)
        reconstruction = torch.mean(reconstruction, dim=0).unsqueeze(0)

        return reconstruction


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()

        self.patch_match = PatchMatch(args.patch_match_args)

        filter_size = args.feature_extractor_filter_size
        self.feature_extractor = feature_extractor(filter_size)
        self.reconstruct = Reconstruct(filter_size)

    def forward(self, left_input, right_input):
        """
        ImageReconstruction:
        Description: This class performs the task of reconstruction the left image using the data of the other image,,
            by fidning correspondences (nnf) between the two fields.
            The images acan be any random images with some overlap between the two to assist
            the correspondence matching.
            For feature_extractor, we just use the RGB features of a (self.filter_size * self.filter_size) patch
            around each pixel.
            For finding the correspondences, we use the Differentiable PatchMatch.
            ** Note: There is no assumption of rectification between the two images. **
            ** Note: The words 'left' and 'right' do not have any significance.**


        Args:
            :left_input:  Left Image (Image 1)
            :right_input:  Right Image (Image 2)

        Returns:
            :reconstruction: Reconstructed left image.
        """

        left_features, right_features, neighbour_extraction_filter = self.feature_extractor(left_input, right_input)
        offset_x, offset_y, x_coordinate, y_coordinate = self.patch_match(left_features, right_features)

        reconstruction = self.reconstruct(right_input,
                                          offset_x, offset_y,
                                          x_coordinate, y_coordinate,
                                          neighbour_extraction_filter.squeeze(1))

        return reconstruction

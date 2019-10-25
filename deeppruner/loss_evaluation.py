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
import torch.nn.functional as F
from collections import namedtuple
import logging

loss_weights = {
    'alpha_super_refined': 1.6,
    'alpha_refined': 1.3,
    'alpha_ca': 1.0,
    'alpha_quantile': 1.0,
    'alpha_min_max': 0.7
}

loss_weights = namedtuple('loss_weights', loss_weights.keys())(*loss_weights.values())

def loss_evaluation(result, disp_true, mask, cost_aggregator_scale=4):

    # forces min_disparity to be equal or slightly lower than the true disparity
    quantile_mask1 = ((disp_true[mask] - result[-1][mask]) < 0).float()
    quantile_loss1 = (disp_true[mask] - result[-1][mask]) * (0.05 - quantile_mask1)
    quantile_min_disparity_loss = quantile_loss1.mean()

    # forces max_disparity to be equal or slightly larger than the true disparity
    quantile_mask2 = ((disp_true[mask] - result[-2][mask]) < 0).float()
    quantile_loss2 = (disp_true[mask] - result[-2][mask]) * (0.95 - quantile_mask2)
    quantile_max_disparity_loss = quantile_loss2.mean()

    min_disparity_loss = F.smooth_l1_loss(result[-1][mask], disp_true[mask], size_average=True)
    max_disparity_loss = F.smooth_l1_loss(result[-2][mask], disp_true[mask], size_average=True)
    ca_depth_loss = F.smooth_l1_loss(result[-3][mask], disp_true[mask], size_average=True)
    refined_depth_loss = F.smooth_l1_loss(result[-4][mask], disp_true[mask], size_average=True)

    logging.info("============== evaluated losses ==================")
    if cost_aggregator_scale == 8:
        refined_depth_loss_1 = F.smooth_l1_loss(result[-5][mask], disp_true[mask], size_average=True)
        loss = (loss_weights.alpha_super_refined * refined_depth_loss_1)
        output_disparity = result[-5]
        logging.info('refined_depth_loss_1: %.6f', refined_depth_loss_1)
    else:
        loss = 0
        output_disparity = result[-4]

    loss += (loss_weights.alpha_refined * refined_depth_loss) + \
            (loss_weights.alpha_ca * ca_depth_loss) + \
            (loss_weights.alpha_quantile * (quantile_max_disparity_loss + quantile_min_disparity_loss)) + \
            (loss_weights.alpha_min_max * (min_disparity_loss + max_disparity_loss))

    logging.info('refined_depth_loss: %.6f' % refined_depth_loss)
    logging.info('ca_depth_loss: %.6f' % ca_depth_loss)
    logging.info('quantile_loss_max_disparity: %.6f' % quantile_max_disparity_loss)
    logging.info('quantile_loss_min_disparity: %.6f' % quantile_min_disparity_loss)
    logging.info('max_disparity_loss: %.6f' % max_disparity_loss)
    logging.info('min_disparity_loss: %.6f' % min_disparity_loss)
    logging.info("==================================================\n")

    return loss, output_disparity

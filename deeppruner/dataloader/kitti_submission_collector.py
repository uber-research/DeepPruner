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
import os


def datacollector(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp = 'disp_occ_0/'

    image = [img for img in sorted(os.listdir(os.path.join(filepath,left_fold))) if img.find('.png') > -1]

    left_test = [os.path.join(filepath, left_fold, img) for img in image]
    right_test = [os.path.join(filepath, right_fold, img) for img in image]

    return left_test, right_test

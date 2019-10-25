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
import torch.utils.data as data
import os
import glob


def datacollector(train_filepath, val_filepath, filepath_2012):

    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'

    left_fold_2012 = 'colored_0/'
    right_fold_2012 = 'colored_1/'
    disp_L_2012 = 'disp_occ/'

    left_train = []
    right_train = []
    disp_train_L = []
    
    left_val = []
    right_val = []
    disp_val_L = []


    if train_filepath is not None:
        left_train = sorted(glob.glob(os.path.join(train_filepath, left_fold, '*.png'))) 
        right_train = sorted(glob.glob(os.path.join(train_filepath, right_fold, '*.png')))
        disp_train_L = sorted(glob.glob(os.path.join(train_filepath, disp_L, '*.png')))

    if filepath_2012 is not None:
        left_train +=sorted(glob.glob(os.path.join(filepath_2012, left_fold_2012, '*_10.png')))
        right_train += sorted(glob.glob(os.path.join(filepath_2012, right_fold_2012, '*_10.png')))
        disp_train_L += sorted(glob.glob(os.path.join(filepath_2012, disp_L_2012, '*_10.png')))

    if val_filepath is not None:
        left_val = sorted(glob.glob(os.path.join(val_filepath, left_fold, '*.png')))
        right_val = sorted(glob.glob(os.path.join(val_filepath, right_fold, '*.png')))
        disp_val_L = sorted(glob.glob(os.path.join(val_filepath, disp_L, '*.png')))

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

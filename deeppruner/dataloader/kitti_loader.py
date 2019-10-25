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
import random
from PIL import Image
import numpy as np
from dataloader import preprocess


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)

# train/ validation image crop size constants
DEFAULT_TRAIN_IMAGE_HEIGHT = 256
DEFAULT_TRAIN_IMAGE_WIDTH = 512

DEFAULT_VAL_IMAGE_HEIGHT = 320
DEFAULT_VAL_IMAGE_WIDTH = 1216


class KITTILoader(data.Dataset):
    def __init__(self, left_images, right_images, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left_img = left_images
        self.right_img = right_images
        self.left_disp = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left_img = self.left_img[index]
        right_img = self.right_img[index]
        left_disp = self.left_disp[index]

        left_img = self.loader(left_img)
        right_img = self.loader(right_img)
        left_disp = self.dploader(left_disp)
        w, h = left_img.size


        if self.training:
            th, tw = DEFAULT_TRAIN_IMAGE_HEIGHT, DEFAULT_TRAIN_IMAGE_WIDTH
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        else:
            th, tw = DEFAULT_VAL_IMAGE_HEIGHT, DEFAULT_VAL_IMAGE_WIDTH
            x1 = w - DEFAULT_VAL_IMAGE_WIDTH
            y1 = h - DEFAULT_VAL_IMAGE_HEIGHT

        left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
        left_disp = left_disp.crop((x1, y1, x1 + tw, y1 + th)) 
        left_disp = np.ascontiguousarray(left_disp, dtype=np.float32) / 256
   

        processed = preprocess.get_transform()
        left_img = processed(left_img)
        right_img = processed(right_img)

        return left_img, right_img, left_disp

    def __len__(self):
        return len(self.left_img)

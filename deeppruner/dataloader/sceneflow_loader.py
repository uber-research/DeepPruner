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
from dataloader import preprocess
from dataloader import readpfm as rp
import numpy as np
import math

# train/ validation image crop size constants
DEFAULT_TRAIN_IMAGE_HEIGHT = 256
DEFAULT_TRAIN_IMAGE_WIDTH = 512

def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class SceneflowLoader(data.Dataset):
    def __init__(self, left_images, right_images, left_disparity, downsample_scale, training, loader=default_loader, dploader=disparity_loader):

        self.left_images = left_images
        self.right_images = right_images
        self.left_disparity = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        # downsample_scale denotes maximum times the image features are downsampled 
        # by the network.
        # Since the image size used for evaluation may not be divisible by the downsample_scale,
        # we pad it with zeros, so that it becomes divible and later unpad the extra zeros.
        self.downsample_scale = downsample_scale

    def __getitem__(self, index):
        left_img = self.left_images[index]
        right_img = self.right_images[index]
        left_disp = self.left_disparity[index]

        left_img = self.loader(left_img)
        right_img = self.loader(right_img)
        left_disp, left_scale = self.dploader(left_disp)
        left_disp = np.ascontiguousarray(left_disp, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = DEFAULT_TRAIN_IMAGE_HEIGHT, DEFAULT_TRAIN_IMAGE_WIDTH

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_disp = left_disp[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, left_disp
        else:
            w, h = left_img.size

            dw = w + (self.downsample_scale - (w%self.downsample_scale + (w%self.downsample_scale==0)*self.downsample_scale))
            dh = h + (self.downsample_scale - (h%self.downsample_scale + (h%self.downsample_scale==0)*self.downsample_scale))

            left_img = left_img.crop((w - dw, h - dh, w, h))
            right_img = right_img.crop((w - dw, h - dh, w, h))

            processed = preprocess.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, left_disp, dw-w, dh-h

    def __len__(self):
        return len(self.left_images)

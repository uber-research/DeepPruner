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
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage.io
import numpy as np
import logging
from dataloader import kitti_submission_collector as ls
from dataloader import preprocess
from PIL import Image
from models.deeppruner import DeepPruner
from models.config import config as config_args
from setup_logging import setup_logging

parser = argparse.ArgumentParser(description='DeepPruner')
parser.add_argument('--datapath', default='/',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--save_dir', default='./',
                    help='save directory')
parser.add_argument('--logging_filename', default='./submission_kitti.log',
                    help='filename for logs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.cost_aggregator_scale = config_args.cost_aggregator_scale
args.downsample_scale = args.cost_aggregator_scale * 8.0

setup_logging(args.logging_filename)

if args.cuda:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


test_left_img, test_right_img = ls.datacollector(args.datapath)

model = DeepPruner()
model = nn.DataParallel(model)
    
if args.cuda:
    model.cuda()

logging.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


if args.loadmodel is not None:
    logging.info("loading model...")
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=True)


def test(imgL, imgR):
    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        refined_disparity = model(imgL, imgR)
        return refined_disparity


def main():

    for left_image_path, right_image_path in zip(test_left_img, test_right_img):
        imgL = np.asarray(Image.open(left_image_path))
        imgR = np.asarray(Image.open(right_image_path))

        processed = preprocess.get_transform()
        imgL = processed(imgL).numpy()
        imgR = processed(imgR).numpy()

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        w = imgL.shape[3]
        h = imgL.shape[2]
        dw = int(args.downsample_scale - (w%args.downsample_scale + (w%args.downsample_scale==0)*args.downsample_scale))
        dh = int(args.downsample_scale - (h%args.downsample_scale + (h%args.downsample_scale==0)*args.downsample_scale))

        top_pad = dh
        left_pad = dw
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        disparity = test(imgL, imgR)
        disparity = disparity[0, top_pad:, :-left_pad].data.cpu().numpy()
        skimage.io.imsave(os.path.join(args.save_dir, left_image_path.split('/')
                                       [-1]), (disparity * 256).astype('uint16'))

        logging.info("Disparity for {} generated at: {}".format(left_image_path, os.path.join(args.save_dir, 
                                                                left_image_path.split('/')[-1])))


if __name__ == '__main__':
    main()

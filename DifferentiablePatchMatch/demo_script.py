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
from PIL import Image
import torch
import random
import skimage
import numpy as np
from models.image_reconstruction import ImageReconstruction
import os
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Differentiable PatchMatch')
parser.add_argument('--base_dir', default='./',
                    help='path of base directory where images are stored.')
parser.add_argument('--save_dir',  default='./',
                     help='save directory')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.backends.cudnn.benchmark=True
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

model = ImageReconstruction()

if args.cuda:
    model.cuda()


def main():
    
    base_dir = args.base_dir
    for file1, file2 in zip(sorted(os.listdir(base_dir+'/image_1')), sorted(os.listdir(base_dir+'/image_2'))):

        image_1_image_path = base_dir + '/image_1/' + file1
        image_2_image_path = base_dir + '/image_2/' + file2

        image_1 = np.asarray(Image.open(image_1_image_path).convert('RGB'))
        image_2 = np.asarray(Image.open(image_2_image_path).convert('RGB'))
        
        image_1 = transforms.ToTensor()(image_1).unsqueeze(0).cuda().float()
        image_2 = transforms.ToTensor()(image_2).unsqueeze(0).cuda().float()

        reconstruction = model(image_1, image_2)

        plt.imsave(os.path.join(args.save_dir, image_1_image_path.split('/')[-1]),
                np.asarray(reconstruction[0].permute(1,2,0).data.cpu()*256).astype('uint16'))



if __name__ == '__main__':
    with torch.no_grad():
        main()
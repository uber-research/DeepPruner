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
from PIL import Image
import os
import os.path
import logging

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath_monkaa, filepath_flying, filepath_driving):

    try:
        monkaa_path = os.path.join(filepath_monkaa, 'monkaa_frames_cleanpass')
        monkaa_disp = os.path.join(filepath_monkaa, 'monkaa_disparity')
        monkaa_dir = os.listdir(monkaa_path)

        all_left_img = []
        all_right_img = []
        all_left_disp = []
        test_left_img = []
        test_right_img = []
        test_left_disp = []

        for dd in monkaa_dir:
            for im in os.listdir(os.path.join(monkaa_path, dd, 'left')):
                if is_image_file(os.path.join(monkaa_path, dd, 'left', im)) and is_image_file(
                        os.path.join(monkaa_path, dd, 'right', im)):
                    all_left_img.append(os.path.join(monkaa_path, dd, 'left', im))
                    all_left_disp.append(os.path.join(monkaa_disp, dd, 'left', im.split(".")[0] + '.pfm'))
                    all_right_img.append(os.path.join(monkaa_path, dd, 'right', im))

    except:
        logging.error("Some error in Monkaa, Monkaa might not be loaded correctly in this case...")
        raise Exception('Monkaa dataset couldn\'t be loaded correctly.')

    
    try:
        flying_path = os.path.join(filepath_flying, 'frames_cleanpass')
        flying_disp = os.path.join(filepath_flying, 'disparity')
        flying_dir = flying_path + '/TRAIN/'
        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))

            for ff in flying:
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff, 'left'))
                for im in imm_l:
                    if is_image_file(os.path.join(flying_dir, ss, ff, 'left', im)):
                        all_left_img.append(os.path.join(flying_dir, ss, ff, 'left', im))

                    all_left_disp.append(os.path.join(flying_disp, 'TRAIN', ss, ff, 'left', im.split(".")[0] + '.pfm'))

                    if is_image_file(os.path.join(flying_dir, ss, ff, 'right', im)):
                        all_right_img.append(os.path.join(flying_dir, ss, ff, 'right', im))

        flying_dir = flying_path + '/TEST/'
        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(os.path.join(flying_dir, ss))

            for ff in flying:
                imm_l = os.listdir(os.path.join(flying_dir, ss, ff, 'left'))
                for im in imm_l:
                    if is_image_file(os.path.join(flying_dir, ss, ff, 'left', im)):
                        test_left_img.append(os.path.join(flying_dir, ss, ff, 'left', im))

                    test_left_disp.append(os.path.join(flying_disp, 'TEST', ss, ff, 'left', im.split(".")[0] + '.pfm'))

                    if is_image_file(os.path.join(flying_dir, ss, ff, 'right', im)):
                        test_right_img.append(os.path.join(flying_dir, ss, ff, 'right', im))
    
    except:
        logging.error("Some error in Flying Things, Flying Things might not be loaded correctly in this case...")
        raise Exception('Flying Things dataset couldn\'t be loaded correctly.')

    try:
        driving_dir = os.path.join(filepath_driving, 'driving_frames_cleanpass/')
        driving_disp = os.path.join(filepath_driving, 'driving_disparity/')

        subdir1 = ['35mm_focallength', '15mm_focallength']
        subdir2 = ['scene_backwards', 'scene_forwards']
        subdir3 = ['fast', 'slow']

        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(os.path.join(driving_dir, i, j, k, 'left'))
                    for im in imm_l:
                        if is_image_file(os.path.join(driving_dir, i, j, k, 'left', im)):
                            all_left_img.append(os.path.join(driving_dir, i, j, k, 'left', im))
                        all_left_disp.append(os.path.join(driving_disp, i, j, k, 'left', im.split(".")[0] + '.pfm'))

                        if is_image_file(os.path.join(driving_dir, i, j, k, 'right', im)):
                            all_right_img.append(os.path.join(driving_dir, i, j, k, 'right', im))
    except:
        logging.error("Some error in Driving, Driving might not be loaded correctly in this case...")
        raise Exception('Driving dataset couldn\'t be loaded correctly.')

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

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
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import skimage
import skimage.transform
import numpy as np
from dataloader import kitti_collector as ls
from dataloader import kitti_loader as DA
from models.deeppruner import DeepPruner
from tensorboardX import SummaryWriter
from torchvision import transforms
from loss_evaluation import loss_evaluation
from models.config import config as config_args
import matplotlib.pyplot as plt
import logging
from setup_logging import setup_logging

parser = argparse.ArgumentParser(description='DeepPruner')
parser.add_argument('--train_datapath_2015', default=None,
                    help='training data path of KITTI 2015')
parser.add_argument('--datapath_2012', default=None,
                    help='data path of KITTI 2012 (all used for training)')
parser.add_argument('--val_datapath_2015', default=None,
                    help='validation data path of KITTI 2015')
parser.add_argument('--epochs', type=int, default=1040,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--logging_filename', default='./finetune_kitti.log',
                    help='filename for logs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


args.cost_aggregator_scale = config_args.cost_aggregator_scale

setup_logging(args.logging_filename)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.datacollector(
    args.train_datapath_2015, args.val_datapath_2015, args.datapath_2012)


TrainImgLoader = torch.utils.data.DataLoader(
         DA.KITTILoader(all_left_img, all_right_img, all_left_disp, True),
         batch_size=16, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.KITTILoader(test_left_img, test_right_img, test_left_disp, False),
         batch_size=8, shuffle=False, num_workers=4, drop_last=False)

model = DeepPruner()
writer = SummaryWriter()
model = nn.DataParallel(model)
    

if args.cuda:
    model.cuda()

if args.loadmodel is not None:
    logging.info("loading model...")
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=True)


logging.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L, iteration, epoch):
    if epoch >= 800:
		model.eval()
    else:	
        model.train()

    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    mask = (disp_true > 0)
    mask.detach_()

    optimizer.zero_grad()
    result = model(imgL,imgR)

    loss, _ = loss_evaluation(result, disp_true, mask, args.cost_aggregator_scale)

    loss.backward()
    optimizer.step()

    return loss.item()

    
    
def test(imgL,imgR,disp_L,iteration):

        model.eval()
        with torch.no_grad():
            imgL   = Variable(torch.FloatTensor(imgL))
            imgR   = Variable(torch.FloatTensor(imgR))   
            disp_L = Variable(torch.FloatTensor(disp_L))

            if args.cuda:
                imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

            mask = (disp_true > 0)
            mask.detach_()
            
            optimizer.zero_grad()
            
            result = model(imgL,imgR)
            loss, output_disparity = loss_evaluation(result, disp_true, mask, args.cost_aggregator_scale)

            #computing 3-px error: (source psmnet)#
            true_disp = disp_true.data.cpu()
            disp_true = true_disp
            pred_disp = output_disparity.data.cpu()

            index = np.argwhere(true_disp>0)
            disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
            correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
            torch.cuda.empty_cache()             
            
            loss = 1-(float(torch.sum(correct))/float(len(index[0])))
                
        return loss


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 500:
        lr = 0.0001
    elif epoch<=1000:
        lr = 0.00005
    else:
        lr = 0.00001
    logging.info('learning rate = %.5f' %(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    for epoch in range(0, args.epochs):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)
    
        if epoch %1==0 and epoch!=0:
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss = test(imgL,imgR,disp_L,batch_idx)
                total_test_loss += test_loss
                logging.info('Iter %d 3-px error in val = %.3f \n' %(batch_idx, test_loss))
                
            logging.info('epoch %d total test loss = %.3f' %(epoch, total_test_loss/len(TestImgLoader)))
            writer.add_scalar("val-loss",total_test_loss/len(TestImgLoader),epoch)
            
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            loss = train(imgL_crop,imgR_crop,disp_crop_L,batch_idx,epoch)
            total_train_loss += loss
            logging.info('Iter %d training loss = %.3f \n' %(batch_idx, loss))
            
        logging.info('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        writer.add_scalar("loss",total_train_loss/len(TrainImgLoader),epoch)

        # SAVE
        if epoch%1==0:
            savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss,
                    'test_loss': total_test_loss,
                }, savefilename)


if __name__ == '__main__':
    main()

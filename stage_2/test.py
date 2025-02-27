from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import Smoke_data as lt
from dataloader import Smoke_loader as DA
from models import *
import torchvision
from PIL import Image, ImageOps,ImageFilter

parser = argparse.ArgumentParser(description='SRNet')
parser.add_argument('--model', default='',
                    help='select model')
parser.add_argument('--datapath', default='',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/results_bunn_fea_rgbd_gradn2_noar_2',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.makedirs(args.savemodel, exist_ok=True)
os.makedirs(args.savemodel+'/att', exist_ok=True)
os.makedirs(args.savemodel+'/res', exist_ok=True)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

datapath1 = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/data/build/test'
all_left_img, all_gt_img, train_index, train_depth = lt.dataloader(datapath1)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img,all_gt_img,train_index,train_depth,False),
    batch_size= 1, shuffle = False,num_workers = 1,drop_last = False)

model = basic()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

modelpath = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/checkpoints_bunn_fea_rgbd_gradn2_noar/checkpoint_800.tar'
pretrain_dict = torch.load(modelpath)
model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

out_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage()
])

def test(tgt_img, tgt_depth, tgt_time, gt_img):
        
        model.eval()
        tgt_img = tgt_img.cuda()
        tgt_time = tgt_time.cuda()
        tgt_depth = tgt_depth.cuda()
        gt_img = gt_img.cuda()

        with torch.no_grad():
           output = model(tgt_img,tgt_depth,tgt_time)
           output = output.squeeze()
        #    att = att.squeeze()
        #    res = res.squeeze()

        #    mask = torch.where(tgt_depth>0.17,1.0,0.0).squeeze(0)
        #    mask2 = torch.where(gt_img>0.03,1.0,0.0).squeeze(0)
        #    mask = mask*mask2
           
        #    output = output*mask
        # return output,att,res
        return output,output,output

def main():
    #------------- TEST ------------------------------------------------------------
    for batch_idx, (tgt_img, tgt_depth, tgt_time, gt_img, top_pad, right_pad) in enumerate(TestImgLoader):
    
        start_time = time.time()
        result,att,res = test(tgt_img, tgt_depth, tgt_time, gt_img)
        print('time = %.2f' %(time.time() - start_time))

        # result = result[:,top_pad:,:-right_pad]
        # att = att[top_pad:,:-right_pad]
        # res = res[:,top_pad:,:-right_pad]
        # result = result[:,top_pad:,:-right_pad]
        # result = torchvision.utils.make_grid(result.detach().cpu(), nrow=1, padding=0, normalize=False) 
        result = (result.detach().cpu()*0.5) + 0.5
        att = (att.detach().cpu()*0.5) + 0.5
        res = (res.detach().cpu()*0.5) + 0.5
        
        # print(torch.mean((att>0.9).float()))
        # att[att<0.9] = 0.0
        # att[att>0.9] = 1.0
        # result = result*att + 1.0*(1-att)
        
        result = out_transform(result)
        att = out_transform(att)
        res = out_transform(res)
        
        # result = result.filter(ImageFilter.MedianFilter(7))
        result.save(args.savemodel+'/test_img_%06d.jpg'%(batch_idx))
        # att.save(args.savemodel+'/att/test_img_%06d.jpg'%(batch_idx))
        # res.save(args.savemodel+'/res/test_img_%06d.jpg'%(batch_idx))
        

if __name__ == '__main__':
   main()







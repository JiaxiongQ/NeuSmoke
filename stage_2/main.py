from __future__ import print_function
import argparse
import os
import random

import torch, torchvision
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torch.autograd import Variable

import numpy as np
import time
import math
from dataloader import Smoke_data as lt
from dataloader import Smoke_loader as DA
from models import *
from torchvision.models import VGG19_Weights
parser = argparse.ArgumentParser(description='SRNet')
parser.add_argument('--model', default='',
                    help='select model')
parser.add_argument('--datapath', default='',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=800,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/checkpoints_bunn_fea_rgbd_gradn2_noar',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.makedirs(args.savemodel, exist_ok=True)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

datapath1 = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s2n/data/build/train'
all_left_img, all_gt_img, train_index, train_depth = lt.dataloader(datapath1)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_gt_img,train_index,train_depth,True), 
         batch_size= 32, shuffle= True, num_workers= 8, drop_last=False)

model = basic()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# VGG Loss Tool
class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.vgg = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

    def forward(self, es, ta):
        
        self.vgg = self.vgg.cuda()
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

        if self.inp_scale == "-11":
            es = (es + 1) / 2
            ta = (ta + 1) / 2
        elif self.inp_scale != "01":
            raise Exception("invalid input scale")
        
        es = (es.cuda() - self.mean) / self.std
        ta = (ta.cuda() - self.mean) / self.std

        loss = [torch.abs(es - ta).mean()]
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return loss

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
    mask_x = torch.mul(mask[:, :, :, 1:], mask[:, :, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
    mask_y = torch.mul(mask[:, :, 1:, :], mask[:, :, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (2, 3)) + torch.sum(grad_y, (2, 3))

    return reduction(image_loss, M)

class GradientLoss(nn.Module):
    def __init__(self, scales=3, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, :, ::step, ::step], target[:, :, ::step, ::step],
                                   mask[:, :, ::step, ::step], reduction=self.__reduction)

        return total

class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the binary focal loss between each element
    in the input and target tensors.

    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean

    Attributes
    -----------
        gamma: float (optional)
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean'):
        if reduction not in ("sum", "mean", "none"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.

        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask

        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        #Warn that if sizes don't match errors may occur
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size"\
                "({input_tensor.size()}). \n This will likely lead to incorrect results"\
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )
        #Broadcast to get sizes/shapes to match
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of binary focal loss
        pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
        pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
        loss_tensor = -(1-pt_tensor)**self.gamma*torch.log(pt_tensor)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            return torch.mean(loss_tensor)
        #If not none or mean, sum
        return torch.sum(loss_tensor)

vgg_loss_func = VGGPerceptualLoss()

grad_loss_func = GradientLoss()

bf_loss_func = BinaryFocalLoss()

def train(tgt_img, tgt_depth, tgt_time, gt_img):
        model.train()
        tgt_img = tgt_img.cuda()
        tgt_time = tgt_time.cuda()
        tgt_depth = tgt_depth.cuda()
        gt_img = gt_img.cuda()
        
        optimizer.zero_grad()

        # output,att = model(tgt_img,tgt_depth,tgt_time)
        output = model(tgt_img,tgt_depth,tgt_time)
        
        loss = F.l1_loss(output, gt_img, reduction='mean')
        
        img_pred = output*0.5+0.5
        img_tgt = gt_img*0.5+0.5
        # att = att*0.5+0.5
        mask = (tgt_depth > 0.05).detach()
        
        # gt_img2 = torch.mean(img_tgt,1,keepdim=True)*mask
        # gt_img2 = F.normalize(gt_img2)
        # loss2 = 0.1*F.smooth_l1_loss(att, gt_img2, reduction='mean')
        # loss2 = 0.1*bf_loss_func(att, mask)
        loss2 = 0.0
        
        loss_vgg = vgg_loss_func(img_pred, img_tgt)
        loss_vgg = 0.5*(loss_vgg[0]+loss_vgg[1]+loss_vgg[2]+loss_vgg[3]+loss_vgg[4])

        mask = mask.expand(gt_img.shape)
        grad_loss = 0.5*grad_loss_func(img_pred,img_tgt,mask)
            
        loss = loss + loss2 + loss_vgg #+ grad_loss

        loss.backward()
        optimizer.step()

        return loss.data, loss2, loss_vgg.data, grad_loss.data

def adjust_learning_rate(optimizer, epoch):
    if epoch < 100:
        lr = 0.001* (0.5 ** (0 // 50))
    elif epoch < 300:
        lr = 0.001* (0.5 ** (50 // 50))
    elif epoch < 600:
        lr = 0.001* (0.5 ** (100 // 50)) 
    else: 
        lr = 0.001* (0.5 ** (150 // 50))    
         
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs+1):
        print('This is %d-th epoch' %(epoch)) 
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

	   ## training ##
        for batch_idx, (tgt_img, tgt_depth, tgt_time, gt_img) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss,loss2,lossvgg,lossg = train(tgt_img, tgt_depth, tgt_time, gt_img)
            print('Iter %d training loss = %.5f, loss2 = %.5f, lossvgg = %.5f, lossg = %.5f, time = %.2f' %(batch_idx, loss, loss2, lossvgg, lossg, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.5f' %(epoch, total_train_loss/len(TrainImgLoader)))

        if epoch % 400 == 0:
            #SAVE
            savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
            }, savefilename)
        
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
    

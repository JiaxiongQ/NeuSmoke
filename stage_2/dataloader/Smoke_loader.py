import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from PIL import Image, ImageOps
import numpy as np
from .preprocess import *
import cv2 as cv
import torchvision
from skimage.util import random_noise

# from .custom_transform import *
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def depth_loader(path):
    depth = cv.imread(path,-1).astype(np.float32) / 60000.0
    # depth = depth*500.0
    return depth

img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
map_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class myImageFloder(data.Dataset):
    def __init__(self, left, gt, index2, depth, training, loader=default_loader, dloader=depth_loader):
 
        self.left = left
        self.gt = gt
        self.depth = depth
        self.loader = loader
        self.dloader = dloader
        self.training = training
        
        steps = []
        
        for i in range(len(index2)):
            steps.append(np.loadtxt(index2[i],dtype=np.float32,delimiter=" "))
        self.steps = np.array(steps)
            
        self._img_transform = img_transform
        self._map_transform = map_transform
        
    def load_img(self, img_path):
        img = self.loader(img_path)
        imgt = self._img_transform(img)
        return imgt
        
    def load_depth(self, depth_path):
        depth = self.dloader(depth_path)
        deptht = self._map_transform(depth)
        return deptht
        
    def __getitem__(self, index):
        left  = self.left[index]
        gt = self.gt[index]
        depth = self.depth[index]
        
        gt_img = self.load_img(gt)
        tgt_img = self.load_img(left)
        tgt_depth = self.load_depth(depth)
        
        # print(gt_img.shape, tgt_img.shape,tgt_depth.shape)
        # exit(0)
        step = np.array(self.steps[index])
        # stept = step.repeat(self.steps.shape[0])
        # step_idxs = np.array(np.where((stept-self.steps)==0.0))[0]
        
        tgt_time = torch.from_numpy(step)
        # ref_imgs = []
        # ref_depths = []
        # ref_intris = []
        # ref_intris_inv = []
        # ref_extris = []
        
        # for i in range(step_idxs.shape[0]):
        #     if step_idxs[i] == index:
        #         continue
        #     ref_imgs.append(self.load_img(self.left[step_idxs[i]]))
        #     ref_depths.append(self.load_depth(self.depth[step_idxs[i]]))
        #     ref_intris.append(torch.from_numpy(self.intris[step_idxs[i]]).float())
        #     ref_intris_inv.append(torch.from_numpy(np.linalg.inv(self.intris[step_idxs[i]])).float())
        #     rel_pose = self.extris[step_idxs[i]] @ tgt_pose
        #     ref_extris.append(torch.from_numpy(rel_pose).float())
        
        if tgt_img.shape[1] % 16 != 0:
            times = tgt_img.shape[1]//16       
            top_pad = (times+1)*16 -tgt_img.shape[1]
        else:
            top_pad = 0

        if tgt_img.shape[2] % 16 != 0:
            times = tgt_img.shape[2]//16                       
            right_pad = (times+1)*16-tgt_img.shape[2]
        else:
            right_pad = 0    

        tgt_img = F.pad(tgt_img,(0,right_pad, top_pad,0))
        gt_img = F.pad(gt_img,(0,right_pad, top_pad,0))
        
        tgt_depth = F.pad(tgt_depth,(0,right_pad, top_pad,0))
        
        if self.training:
            th, tw = 96,128
            c,h,w= tgt_img.shape
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            
            tgt_img = tgt_img[:, y1:y1 + th, x1:x1 + tw]
            tgt_depth = tgt_depth[:, y1:y1 + th, x1:x1 + tw]
            gt_img = gt_img[:, y1:y1 + th, x1:x1 + tw]
        

        if self.training:
            return tgt_img, tgt_depth, tgt_time, gt_img
        else:
            return tgt_img, tgt_depth, tgt_time, gt_img, top_pad, right_pad

    def __len__(self):
        return len(self.left)

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
    def __init__(self, left, depth, training, loader=default_loader, dloader=depth_loader):
 
        self.left = left
        self.depth = depth
        self.loader = loader
        self.dloader = dloader
        self.training = training
        
        steps = torch.linspace(0., 1., 60 + 1)
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
        depth = self.depth[index]
        
        tgt_img = self.load_img(left)
        tgt_depth = self.load_depth(depth)
        
        tgt_time = self.steps[index]
        
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
        
        tgt_depth = F.pad(tgt_depth,(0,right_pad, top_pad,0))

        return tgt_img, tgt_depth, tgt_time, top_pad, right_pad

    def __len__(self):
        return len(self.left)

import torch.utils.data as data
import os
from glob import glob
from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(gtpath):

    train_gt_list = sorted(glob(os.path.join(gtpath, 'gt_2/*.png')))
    
    train_in_list = sorted(glob(os.path.join(gtpath, 'imgs/*.png')))
    
    train_gt_depth = sorted(glob(os.path.join(gtpath, 'depth/*.png')))
    
    train_gt_index = sorted(glob(os.path.join(gtpath, 'steps/*.txt')))
    
    
    
    # train_intris = sorted(glob(os.path.join(gtpath, 'intris/*.txt')))
    
    # train_extris = sorted(glob(os.path.join(gtpath, 'extris/*.txt')))
    
    print(len(train_gt_list),len(train_in_list),len(train_gt_index),len(train_gt_depth))
    
    return train_in_list, train_gt_list, train_gt_index, train_gt_depth

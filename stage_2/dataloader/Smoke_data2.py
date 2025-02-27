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

    all_img_lis = sorted(glob(os.path.join(gtpath, 'imgs/*.png')))
    
    train_gt_depth = sorted(glob(os.path.join(gtpath, 'depth/*.png')))
    
    print(len(all_img_lis),len(train_gt_depth))
    
    return all_img_lis,train_gt_depth

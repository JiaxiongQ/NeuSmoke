import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import math
from . import utils

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R

# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w

#     c2w = torch.Tensor(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w

def pose_spherical(theta, phi, radius, rotZ=True, wx=0.0, wy=0.0, wz=0.0):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # wx,wy,wz, additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    if rotZ: # swap yz, and keep right-hand
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w

    ct = torch.Tensor([
        [1,0,0,wx],
        [0,1,0,wy],
        [0,0,1,wz],
        [0,0,0,1]]).float()
    c2w = ct @ c2w
    
    return c2w

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[0, 0] = -1.0
    flip_yz[2, 2] = -1.0
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def load_dnerf_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    # translate = [-0.3382070094283088, 
    #              -0.38795384153014023, 
    #              0.2609209839653898]
    # scale = 0.6739255445645317
    
    # translate = [0.795, 
    #             0.551, 
    #             0.0]
    # scale = 0.05314434680887337
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        skip = testskip
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            
            img = imageio.imread(fname)
            imgs.append(img)
            
            pt = np.array(frame['transform_matrix']).reshape((4, 4)) 
            # cam_center = pt[:3, 3]
            # cam_center = (cam_center + translate) * scale
            # pt[:3, 3] = cam_center  
            # pose = convert_pose(pt)      
            pose = pt
            
            poses.append(pose)
            cur_time = frame['time'] #if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
            if cur_time==0.0:
                W2C = np.linalg.inv(pose)
                print(W2C[:3,3])
            times.append(cur_time)
        
        assert times[0] == 0, "Time must start at 0"
        # exit(0)
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    camera_angle_x = float(math.pi/4)
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # print(focal)
    # exit(0)
    # if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
    #     with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
    #         meta = json.load(fp)
    #     render_poses = []
    #     for frame in meta['frames']:
    #         render_poses.append(np.array(frame['transform_matrix']))
    #     render_poses = np.array(render_poses).astype(np.float32)
    # else:
    #     render_poses = torch.stack([pose_spherical(angle, 0.0, 12.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    # render_times = torch.linspace(0., 1., render_poses.shape[0])
    sp_n = 60 # an even number!
    radius = 0.9
    phi = 20
    r_center = [0.3382070094283088, 0.3, -0.2609209839653898]
    sp_poses = [
        pose_spherical(angle, phi, radius, False, r_center[0], r_center[1], r_center[2]) 
        for angle in np.linspace(-180,180,sp_n+1)[:-1]
    ]
    render_poses = torch.stack(sp_poses,0) # [sp_poses[36]]*sp_n, for testing a single pose
    render_times = torch.linspace(0., 1., sp_n)
    
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W,H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split


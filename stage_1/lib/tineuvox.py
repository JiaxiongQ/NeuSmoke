import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo

import cv2

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()
        self._pt, self._pt_out = self.create_net_pt()
        self.act = Sine()
        # self.act = nn.ReLU(inplace=True)
        self.rigidity_tanh = nn.Tanh()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        first_layer_sine_init(layers[0])
        layers[0] = nn.utils.weight_norm(layers[0])
        
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
            
            sine_init(layers[-1])
            layers[-1] = nn.utils.weight_norm(layers[-1])
            
        return nn.ModuleList(layers), nn.Linear(self.W, 3)
    
    def create_net_pt(self):
        layers = [nn.Linear(3, self.W)]
        first_layer_sine_init(layers[0])
        layers[0] = nn.utils.weight_norm(layers[0])
        
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
            
            sine_init(layers[-1])
            layers[-1] = nn.utils.weight_norm(layers[-1])
            
        return nn.ModuleList(layers), nn.Linear(self.W, 1)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = self.act(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)
    
    def query_pt(self, new_pts, net, net_final):
        h = new_pts
        for i, l in enumerate(net):
            h = net[i](h)
            h = self.act(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)

    def forward(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        mask = self.query_pt(input_pts[:, :3], self._pt, self._pt_out)
        
        input_pts_orig = input_pts[:, :3]
        rigidity_mask = (self.rigidity_tanh(mask) + 1.0) / 2.0

        # rigidity_mask[rigidity_mask < 0.9] = 0.0
        out = input_pts_orig + rigidity_mask*dx
        # out = input_pts_orig + dx
        return out

# Model
class RGBNet(nn.Module):
    def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
        """ 
        """
        super(RGBNet, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = h_ch
        self.input_ch_views = views_ch
        self.input_ch_pts = pts_ch
        self.input_ch_times = times_ch
        self.output_ch = output_ch
        self.feature_linears = nn.Linear(self.input_ch, W)
        self.views_linears = nn.Sequential(nn.Linear(W, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
        
    def forward(self, input_h, input_views):
        feature = self.feature_linears(input_h)
        # feature_views = torch.cat([feature, input_views],dim=-1)
        feature_views = feature
        outputs = self.views_linears(feature_views)
        return outputs

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)
    
    def attention(self,q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)#attention
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, v)
        return output,scores
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        concat,score = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = concat.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output,score
    
class SelfAtt(torch.nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.attfunc = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        
    def forward(self, fea):
        h0 = fea
        h1,s = self.attfunc(h0,h0,h0)
        h = h1.squeeze()
        h = h0 + self.dropout1(h)
        return h,s

'''Model'''
class TiNeuVox(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, add_cam=False,
                 alpha_init=None, fast_color_thres=0,
                 voxel_dim=0, defor_depth=3, net_width=128,
                 posbase_pe=10, viewbase_pe=4, timebase_pe=8, gridbase_pe=2,
                 **kwargs):
        
        super(TiNeuVox, self).__init__()
        self.add_cam = add_cam
        self.voxel_dim = voxel_dim
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.posbase_pe = posbase_pe
        self.viewbase_pe = viewbase_pe
        self.timebase_pe = timebase_pe
        self.gridbase_pe = gridbase_pe
        self.times = None
        self.times_feature = None
        self.ray_pts = None
        self.ray_pts_delta = None
        self.rat_idxs = None
        self.tf0 = None
        
        times_ch = 2*timebase_pe+1
        views_ch = 3+3*viewbase_pe*2
        pts_ch = 3+3*posbase_pe*2,
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('TiNeuVox: set density bias shift to', self.act_shift)

        timenet_width = net_width
        timenet_depth = 1
        timenet_outputs = [1,10,20,voxel_dim+voxel_dim*2*gridbase_pe,40,50,60]
        
        timenet_output = timenet_outputs[3]

        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
        nn.Linear(timenet_width, timenet_output))
        if self.add_cam == True:
            views_ch = 3+3*viewbase_pe*2+timenet_output
            self.camnet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output))
            print('TiNeuVox: camnet', self.camnet)

        featurenet_width = net_width
        featurenet_depth = 1
        grid_dim = voxel_dim*3+voxel_dim*3*2*gridbase_pe
        input_dim = grid_dim+timenet_output+0+0+3+3*posbase_pe*2
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
        self.featurenet_width = featurenet_width
        self._set_grid_resolution(num_voxels)
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output)
        input_dim = featurenet_width
        
        # self.compress = nn.Sequential(nn.Linear(grid_dim+timenet_output, 32), nn.ReLU(inplace=True))
        # self.selfatt = SelfAtt(4,32)
        
        self.decoder = nn.Linear(featurenet_width, 3)
        self.decoder2 = nn.Linear(featurenet_width, 3)
        self.relu_act = nn.ReLU(inplace=True)
        
        self.tran_mask = None
        self.vt_featuresc = None
        
        self.densitynet = nn.Linear(input_dim, 1)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(gridbase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))

        self.voxel_dim = voxel_dim
        self.feature= torch.nn.Parameter(torch.zeros([1, self.voxel_dim, *self.world_size],dtype=torch.float32))
        self.rgbnet = RGBNet(W=net_width-1, h_ch=featurenet_width, views_ch=views_ch, pts_ch=pts_ch, times_ch=times_ch, output_ch=featurenet_width)
        
        print('TiNeuVox: feature voxel grid', self.feature.shape)
        print('TiNeuVox: timenet mlp', self.timenet)
        print('TiNeuVox: deformation_net mlp', self.deformation_net)
        print('TiNeuVox: densitynet mlp', self.densitynet)
        print('TiNeuVox: featurenet mlp', self.featurenet)
        print('TiNeuVox: rgbnet mlp', self.rgbnet)


    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('TiNeuVox: voxel_size      ', self.voxel_size)
        print('TiNeuVox: world_size      ', self.world_size)
        print('TiNeuVox: voxel_size_base ', self.voxel_size_base)
        print('TiNeuVox: voxel_size_ratio', self.voxel_size_ratio)


    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'voxel_dim':self.voxel_dim,
            'defor_depth':self.defor_depth,
            'net_width':self.net_width,
            'posbase_pe':self.posbase_pe,
            'viewbase_pe':self.viewbase_pe,
            'timebase_pe':self.timebase_pe,
            'gridbase_pe':self.gridbase_pe,
            'add_cam': self.add_cam,
        }


    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('TiNeuVox: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('TiNeuVox: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.feature = torch.nn.Parameter(
            F.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
 
    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def mult_dist_interp(self, ray_pts_delta):

        x_pad = math.ceil((self.feature.shape[2]-1)/4.0)*4-self.feature.shape[2]+1
        y_pad = math.ceil((self.feature.shape[3]-1)/4.0)*4-self.feature.shape[3]+1
        z_pad = math.ceil((self.feature.shape[4]-1)/4.0)*4-self.feature.shape[4]+1
        grid = F.pad(self.feature.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        vox_l = self.grid_sampler(ray_pts_delta, grid)
        vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
        vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        
        vox_feature = torch.cat((vox_l,vox_m,vox_s),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten

    def activate_density(self, density, interval=None): 
        interval = interval if interval is not None else self.voxel_size_ratio 
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval) 

    def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id,mask_inbbox
    
    def gradient_loss(self,):
        vox_feature_flatten=self.mult_dist_interp(self.ray_pts_delta)
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        # pts deformation 
        ray_pts = self.ray_pts
        ray_pts.requires_grad_(True)
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        
        # ray_id = self.rat_idxs
        # times_sel = self.times[ray_id]
        # times_sel.requires_grad_(True)
        # times_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(times_emb)
        # times_feature = times_feature[ray_id]
        
        times_feature = self.times_feature
        h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        
        density_result = self.densitynet(h_feature)
        
        e1 = torch.ones_like(density_result, device=density_result.get_device())
        e2 = torch.ones_like(density_result, device=density_result.get_device())
        
        e_dt = torch.autograd.grad(density_result, times_feature, 
                                     grad_outputs=e1, 
                                     create_graph=True, 
                                     retain_graph=True)[0]
        
        e_ds = torch.autograd.grad(density_result, ray_pts, 
                                     grad_outputs=e2, 
                                     create_graph=True, 
                                     retain_graph=True)[0]
        
        e_dt = torch.mean(e_dt, dim=-1).unsqueeze(-1)
        
        vel_loss = torch.mean(torch.abs((e_dt + self.ray_pts_delta*e_ds).sum(dim=-1)))

        return vel_loss,e_ds

    def density_func(self, ray_pts, times_sel, stepsize):
        self.times = times_sel
        times_emb = poc_fre(times_sel, self.time_poc)
        times_feature = self.timenet(times_emb)

        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        interval = stepsize * self.voxel_size_ratio
        self.ray_pts = ray_pts
        # pts deformation 
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature)
        self.ray_pts_delta = ray_pts_delta
        
        # voxel query interp
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))  
        
        density_result = self.densitynet(h_feature)
        density_result = F.softplus(density_result + self.act_shift)
        
        return density_result
    
    def extract_density(self, test_times, stepsize, testsavedird):
        N = 64
        resolution = 256
        
        bound_min = self.xyz_min
        bound_max = self.xyz_max
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
        
        with torch.no_grad():
            for idx in range(len(test_times)):
                print(idx)
                u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = torch.meshgrid(xs, ys, zs)
                            pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                            time_one = test_times[idx]*torch.ones_like(pts[:,0:1])
                            den = self.density_func(pts,time_one,stepsize).reshape(len(xs), len(ys), len(zs)).float().detach().cpu().numpy()
                            u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = den
                            
                np.savez_compressed(testsavedird+'/%04d.npz'%idx, dmap = u)
        
    def extract_flow(self, test_times, stepsize, testsavedird):
        
        resolution = 128
        
        import vtk
        from vtk.numpy_interface import algorithms as algs
        from vtk.numpy_interface import dataset_adapter as dsa
        def point2vtk(vectors,filename):
            xs, ys, zs = np.meshgrid(np.arange(resolution), np.arange(resolution), np.arange(resolution))

            polydata = vtk.vtkPolyData()

            pts = vtk.vtkPoints()
            points = algs.make_vector(xs.ravel(),
                                    ys.ravel(),
                                    zs.ravel())
            pts.SetData(dsa.numpyTovtkDataArray(points, "Points"))

            polydata.SetPoints(pts)

            vectors = algs.make_vector(vectors[:,:,:,0].ravel(),
                                    vectors[:,:,:,1].ravel(),
                                    vectors[:,:,:,2].ravel())
            polydata.GetPointData().SetScalars(dsa.numpyTovtkDataArray(vectors, "Velocity"))
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(filename)
            writer.SetInputData(polydata)
            writer.Update()
        
        # import pandas as pd 
        # from pyntcloud as pc
        # def point2ply(pts,vels,filename):
        #     data = {'x':pts[:,0],
        #             'y':pts[:,1],
        #             'z':pts[:,2],
        #             'u':vels[:,0],
        #             'v':vels[:,1],
        #             'w':vels[:,2]
        #            }
        #     cloud = pc(pd.DataFrame(data))
        #     cloud.to_file(filename)          
        N = 64  
        bound_min = self.xyz_min
        bound_max = self.xyz_max
        m_bound = 0.5 * (bound_min + bound_max)
        X = torch.linspace(bound_min[0]*0.5, bound_max[0]*0.5, resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
        
        dt = torch.ones([resolution, resolution, resolution, 1])
        dx = (bound_max[0] - bound_min[0]) / (resolution-1)
        dx = dx*dt
        dy = (bound_max[1] - bound_min[1]) / (resolution-1)
        dy = dy*dt
        dz = (bound_max[2] - bound_min[2]) / (resolution-1)
        dz = dz*dt
        ds = torch.cat([dx,dy,dz], dim=-1).float().detach().cpu().numpy()
        
        with torch.no_grad():
            for idx in range(len(test_times)):
                if idx < 100:
                    continue
                
                u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
                v = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = torch.meshgrid(xs, ys, zs)
                            pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                            
                            time_one = test_times[idx]*torch.ones_like(pts[:,0:1])
                            den = self.density_func(pts,time_one,stepsize).reshape(len(xs), len(ys), len(zs)).float().detach().cpu().numpy()
                            u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = den
                            
                            times_emb = poc_fre(time_one, self.time_poc)
                            times_feature = self.timenet(times_emb)
                            rays_pts_emb = poc_fre(pts, self.pos_poc)
                            
                            ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature)
                            vel = (ray_pts_delta - rays_pts_emb[...,:3]).reshape(len(xs), len(ys), len(zs), 3)
   
                            vel = vel / torch.linalg.norm(vel, ord=2, dim=-1, keepdim=True)
                            v[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = vel.float().detach().cpu().numpy()
                
                print(u.max(),u.min())
                mask = (u>0.01).astype(np.float32)[...,None]
                v = v*mask
                out_file = os.path.join(testsavedird, '{:03d}.vtk'.format(idx))            
                point2vtk(v,out_file)
                exit(0)

    def div_loss(self):
        torch.set_grad_enabled(True)
        ray_pts = self.ray_pts
        ray_pts.requires_grad_(True)
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        
        ray_id = self.rat_idxs
        times_sel = self.times[ray_id]
        times_sel.requires_grad_(True)
        times_emb = poc_fre(times_sel, self.time_poc)
        times_feature = self.timenet(times_emb)
        times_feature = times_feature[ray_id]
        
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature)
        
        e3 = torch.ones_like(ray_pts_delta, device=self.ray_pts_delta.get_device())
        e_ds2 = torch.autograd.grad(ray_pts_delta, ray_pts, 
                                     grad_outputs=e3, 
                                     create_graph=True, 
                                     retain_graph=True)[0]
        torch.set_grad_enabled(False)
        return e_ds2
                
    def forward(self, rays_o, rays_d, viewdirs,times_sel, cam_sel=None,bg_points_sel=None,global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)
        self.times = times_sel
        times_emb = poc_fre(times_sel, self.time_poc)
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        times_feature = self.timenet(times_emb)

        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
            
        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        self.rat_idxs = ray_id
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        self.ray_pts = ray_pts
        # pts deformation 
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)

        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id])
        # bound_min = self.xyz_min
        # bound_max = self.xyz_max
        # scale_ii = [0.3,0.3,0.9]
        # mask = torch.ones_like(ray_pts_delta[:,0])
        # for ii in range(3):
        #     mask *= ((ray_pts[:,ii] > bound_min[ii]*scale_ii[ii]).float() * (ray_pts[:,ii] < bound_max[ii]*scale_ii[ii]).float())  
        # d_pt = ray_pts_delta - ray_pts
        # m_bound = 0.5*(bound_max + bound_min)
        # mask1 = mask*((ray_pts[:,-1] > m_bound[-1]).float())
        # mask2 = mask*((ray_pts[:,-1] <= m_bound[-1]).float())
        # ray_pts_delta[:,0] = ray_pts_delta[:,0] - d_pt[:,0]*0.5*mask1
        # ray_pts_delta[:,0] = ray_pts_delta[:,0] - d_pt[:,0]*0.3*mask2
        # ray_pts_delta[:,1] = ray_pts_delta[:,1] - d_pt[:,1]*0.3*mask1
        # ray_pts_delta[:,1] = ray_pts_delta[:,1] - d_pt[:,1]*0.1*mask2
        self.ray_pts_delta = ray_pts_delta
        # div = self.div_loss()
        
        # computer bg_points_delta
        if bg_points_sel is not None:
            bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
            bg_points_sel_delta = self.deformation_net(bg_points_sel_emb, times_feature[:(bg_points_sel_emb.shape[0])])
            ret_dict.update({'bg_points_delta': bg_points_sel_delta})
        # voxel query interp
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        times_feature = times_feature[ray_id]
        self.times_feature = times_feature
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        # vt_features = torch.cat([vox_feature_flatten_emb,times_feature],dim=-1)
        # vt_featuresc = self.compress(vt_features)
        # vt_featuresc,score = self.selfatt(vt_featuresc)
        # score = score.squeeze()
        
        h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))  
        
        density_result = self.densitynet(h_feature)
        
        alpha = self.activate_density(density_result,interval)
        alpha = alpha.squeeze(-1)
        # tran_mask = (torch.mean(score,dim=-1)>0.7)
        
        # self.tran_mask = tran_mask
        # self.vt_featuresc = vt_features

        if self.fast_color_thres > 0:
            # mask = (alpha > self.fast_color_thres) & tran_mask
            mask = (alpha > self.fast_color_thres)
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            h_feature=h_feature[mask]
        
        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            h_feature=h_feature[mask]
        
        viewdirs_emb_reshape = viewdirs_emb[ray_id]
        if self.add_cam == True:
            viewdirs_emb_reshape=torch.cat((viewdirs_emb_reshape, cams_feature[ray_id]), -1)
        
        rgb_logit0 = self.rgbnet(h_feature, viewdirs_emb_reshape)
        rgb_feature0 = self.relu_act(rgb_logit0)
        rgb_value = self.decoder(rgb_feature0)
        rgb = torch.sigmoid(rgb_value)
    
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')

        rgb_feature = segment_coo(
                src=(weights.unsqueeze(-1)*rgb_feature0),
                index=ray_id,
                out=torch.zeros([N, rgb_feature0.shape[-1]]),
                reduce='sum')
        rgb_logit = self.decoder2(rgb_feature.squeeze())
        rgb_marched2 = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        # rgb_marched2 += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        acc = segment_coo(
                src=(weights.unsqueeze(-1)),
                index=ray_id,
                out=torch.zeros([N, 1]),
                reduce='sum')

        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'acc': acc,
            'rgb_marched': rgb_marched,
            'rgb_marched2': rgb_marched2,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'pts_o': self.ray_pts,
            'pts_delta': self.ray_pts_delta,
        })
        
        with torch.no_grad():
            depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum')
        ret_dict.update({'depth': depth})
        return ret_dict

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@torch.no_grad()
def get_training_rays(rgb_tr, times,train_poses, HW, Ks, ndc):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    times_tr = torch.ones([len(rgb_tr), H, W, 1], device=rgb_tr.device)

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        times_tr[i] = times_tr[i]*times[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, times,train_poses, HW, Ks, ndc):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr=torch.ones([N,1], device=DEVICE)
    times=times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        n = H * W
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, times,train_poses, HW, Ks, ndc, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr = torch.ones([N,1], device=DEVICE)
    times = times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb

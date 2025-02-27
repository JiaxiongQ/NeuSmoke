import torch
import torch.nn as nn
import torch.nn.functional as F
from .inverse_warp import *
from .submodule2D import *

def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    


class SRNet(nn.Module):
    def __init__(self, bending_latent_size=32, ngf=32, groups = 16):
        super(SRNet, self).__init__()
        
        self.groups = groups
        dim = ngf
        sinu_pos_emb = SinusoidalPosEmb(dim)
        time_dim = dim * 4
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.in1_conv1 = self.inconv_w(1, ngf)
        self.in1_down1 = self.down2x(ngf, ngf*2)
        self.in1_down2 = self.down2x(ngf*2, ngf*4)
        self.in1_down3 = self.down2x(ngf*4, ngf*8)
        self.in1_down4 = self.down2x(ngf*8, ngf*16)
        
        self.out_rc0 = ResnetBlock2(ngf*16,ngf*16)
        self.out_up1 = self.up2x(ngf*16, ngf*8)
        self.out_rc1 = ResnetBlock2(ngf*8,ngf*8)
        self.out_up2 = self.up2x(ngf*8, ngf*4)
        self.out_rc2 = ResnetBlock2(ngf*4,ngf*4)
        self.out_up3 = self.up2x(ngf*4, ngf*2)
        self.out_rc3 = ResnetBlock2(ngf*2,ngf*2)
        self.out_up4 = self.up2x(ngf*2, ngf)

        self.out_convs = self.outconv(ngf)
        self.out_convi = nn.Sequential(conv1x1(ngf, 3),
                                       nn.Tanh())
        self.out_convim = nn.Sequential(conv1x1(ngf, 1),
                                       nn.Tanh())
        
        self.out_convc = conv1x1(ngf, 1)
        
    def inconv_w(self, in_channels, out_channels):
        return nn.Sequential(
            WeightStandardizedConv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(self.groups, out_channels),
            nn.SiLU()
        )
    
    def inconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(self.groups, out_channels),
            nn.SiLU()
        )
    
    def down2x(self, in_channels, out_channels):
        return nn.Sequential(
            downconv2x(in_channels, out_channels),
            nn.GroupNorm(self.groups, out_channels),
            nn.SiLU(),
            ResnetBlock(out_channels)
        )
    
    def up2x(self, in_channels, out_channels):
        return nn.Sequential(
            upconv2x(in_channels, out_channels),
            nn.GroupNorm(self.groups, out_channels),
            nn.SiLU()
        )
    
    def outconv(self, in_channels):
        return nn.Sequential(
            ResnetBlock(in_channels),
            ResnetBlock(in_channels)
        )

    def forward(self, tgt_img,tgt_depth,tgt_time,ref_imgs,ref_depths,ref_poses,intrinsics,intrinsics_inv):
        
        confs = []
        predis = []
        
        time_fea = self.time_mlp(tgt_time)
        
        for i,ref_depth in enumerate(ref_depths):
            
            x1 = self.in1_conv1(ref_depth)
            x2 = self.in1_down1(x1)
            x3 = self.in1_down2(x2)
            x4 = self.in1_down3(x3)
            x5 = self.in1_down4(x4)
            
            y = self.out_rc0(x5,time_fea)
            y = self.out_up1(y) + x4
            y = self.out_rc1(y,time_fea) 
            y = self.out_up2(y) + x3
            y = self.out_rc2(y,time_fea)
            y = self.out_up3(y) + x2
            y = self.out_rc3(y,time_fea) 
            y = self.out_up4(y) + x1
            
            depth = ref_depths[i]           
            yf = inverse_warp(y, depth.squeeze(1), ref_poses[i][:,:3,:4], intrinsics[:,:3,:3], intrinsics_inv[:,:3,:3])
                      
            yf = self.out_convs(y)
            conf = self.out_convc(yf)
            confs.append(conf)
            
            im = self.out_convim(yf)
            predi = self.out_convi(yf) + tgt_img.detach()*im
            predis.append(predi)
            
        rgbs = torch.stack(predis)
        alphas = torch.stack(confs)
        alphas = torch.softmax(alphas, dim=0)
        pred = (alphas * rgbs).sum(dim=0)
        
        return pred
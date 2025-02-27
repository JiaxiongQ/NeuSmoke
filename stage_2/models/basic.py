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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.SiLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        avg_out = torch.mean(x0, dim=1, keepdim=True)
        max_out, _ = torch.max(x0, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        x = self.conv1(x)
        return self.sigmoid(x)*x0

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
        
        self.in1_conv1 = self.inconv_w(4, ngf)
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
        
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(ngf)
        
        self.out_convi = nn.Sequential(conv1x1(ngf, 3),
                                       nn.Tanh())
        # self.out_convim = nn.Sequential(conv1x1(ngf+1, 1),
        #                                nn.Tanh())
        
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

    def forward(self, tgt_img, tgt_depth, tgt_time):
        
        confs = []
        predis = []
        
        time_fea = self.time_mlp(tgt_time)
            
        rgbd = torch.cat([tgt_img,tgt_depth],1)  
          
        x1 = self.in1_conv1(rgbd)
        x2 = self.in1_down1(x1)
        x3 = self.in1_down2(x2)
        x4 = self.in1_down3(x3)
        x5 = self.in1_down4(x4)
        
        # y = self.out_rc0(x5)
        # y = self.out_up1(y) + x4
        # y = self.out_rc1(y) 
        # y = self.out_up2(y) + x3
        # y = self.out_rc2(y)
        # y = self.out_up3(y) + x2
        # y = self.out_rc3(y) 
        # y = self.out_up4(y) + x1
        
        y = self.out_rc0(x5,time_fea)
        y = self.out_up1(y) + x4
        y = self.out_rc1(y,time_fea) 
        y = self.out_up2(y) + x3
        y = self.out_rc2(y,time_fea)
        y = self.out_up3(y) + x2
        y = self.out_rc3(y,time_fea) 
        y = self.out_up4(y) + x1
                    
        yf = self.out_convs(y)
        yf = self.sa(yf)
        yf2 = self.ca(yf)
        
        # yf3 = torch.cat([yf,tgt_depth],1)
        # im = self.out_convim(yf3)
        # res = self.out_convi(yf2)
        # pred = 0.5*(res + tgt_img.detach()*im)
        pred = self.out_convi(yf2)
        
        return pred#,im,res
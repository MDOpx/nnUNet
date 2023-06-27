import torch 
import torch.nn as nn
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple=(3, 3, 3), stride0: tuple=(1, 1, 1), stride1: tuple=(1, 1, 1), padding: tuple=(0, 0, 0), bias: bool=True) -> None:
        super().__init__()
        
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size, stride0, padding, bias=bias)
        self.instnorm0 = InstanceNorm3d(out_channels, affine=True)
        self.act0 = LeakyReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size, stride1, padding, bias=bias)
        self.instnorm1 = InstanceNorm3d(out_channels, affine=True)
        self.act1 = LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm0(x)
        x = self.act0(x)

        x = self.conv1(x)
        # if self.dropout is not None:
            # x = self.dropout(x)
        x = self.instnorm1(x)
        x = self.act1(x)
        return x


class CNN_3D_encoder(nn.Module):
    # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
    # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
    # can be expensive, so it makes sense to save and reuse them.
    _gaussian_3d = _patch_size_for_gaussian_3d = None

    def __init__(
        self,
        num_classes: int,
        in_channels: int=1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # For origin nnUNet v1 backward compatibility
        self.do_ds = True
        self.final_nonlin = lambda x: x
        self.deep_supervision = True

        # Encoder phase
        self.encoders = nn.ModuleList([
            Conv3dBlock(in_channels,  32, padding=1),
            Conv3dBlock(         32,  64, padding=1, stride0=2, stride1=1),
            Conv3dBlock(         64, 128, padding=1, stride0=2, stride1=1),
            Conv3dBlock(        128, 256, padding=1, stride0=2, stride1=1),
            Conv3dBlock(        256, 320, padding=1, stride0=2, stride1=1),
            Conv3dBlock(320, 320, kernel_size=3, padding=1, stride0=(1, 2, 2), stride1=1) #Bottle_neck
        ])
        
        self.final_output = 320
    
    def forward(self, x):
        skips = []

        x_enc0 = self.encoders[0](x) # x.shape = 1, 1, 80, 160, 192 
        skips.append(x_enc0)
        x_enc1 = self.encoders[1](x_enc0) #x_enc0.shape = 1, 32, 80, 160, 192
        skips.append(x_enc1)
        x_enc2 = self.encoders[2](x_enc1) #x_enc1.shape = 1, 64, 40, 80, 96
        skips.append(x_enc2)
        x_enc3 = self.encoders[3](x_enc2) #x_enc2.shape = 1, 128, 20, 40, 48
        skips.append(x_enc3)
        x_enc4 = self.encoders[4](x_enc3) #x_enc3.shape = 1, 256, 10, 20, 24
        skips.append(x_enc4)
        x_enc5 = self.encoders[5](x_enc4) #x_enc4.shape = 1, 320, 5, 10, 12
        skips.append(x_enc5)
        
        return x_enc5, skips #x_enc5.shape=1, 320, 5, 5, 6
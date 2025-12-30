import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import inspector API
# 
# Note:
# You can ignore warning message related with XIR. 
# The inspector relies on 'vai_utf' package. In conda env vitis-ai-pytorch in Vitis-AI docker, vai_utf is ready. But if vai_q_pytorch is installed by source code, it needs to install vai_utf in advance.
from pytorch_nndct.apis import Inspector
class DoubleConv(nn.Module):
    """ [Conv2d => ReLU] x2 """
    def __init__(self, in_ch, out_ch, k_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# (1, 128, 1024) -> (1, 64, 64)   
class Unet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__() # Initialize the parent class
        self.target_height = 128
        self.target_width = 128

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Skip connections
        self.skip2  = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,4), stride=(1,4), padding=0),
        )
        self.skip3  = nn.Sequential(    # because stride 8 not supported by DPU
            nn.Conv2d(128, 128, kernel_size=(1,4), stride=(1,4), padding=0),
        )
        self.skip4  = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1,4), stride=(1,4), padding=0),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        # Output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1p = self.pool(e1)
        e2 = self.enc2(e1p) 
        e2p = self.pool(e2) 
        e3 = self.enc3(e2p)
        e3p = self.pool(e3)
        e4 = self.enc4(e3p)
        e4p = self.pool(e4)
        # Bottleneck
        b = self.bottleneck(e4p) # 256, 16, 64 -> 256, 16, 16
        # Skip connections with convolution for resizing
        s4 = self.skip4(e4)  # 256, 32, 128 -> 256, 32, 32 
        s3 = self.skip3(e3)
        s2 = self.skip2(e2)
        # Decoder
        d4 = self.up4(b) # 256, 16, 16 -> 256, 32, 32
        d4c = torch.cat([s4, d4], dim=1)  # Concatenate along the channel dimension
        d4d = self.dec4(d4c)  # 512, 32, 32 -> 128, 32, 32

        d3 = self.up3(d4d)
        d3c = torch.cat([s3, d3], dim=1)
        d3d = self.dec3(d3c)  
        
        d2 = self.up2(d3d)
        d2c = torch.cat([s2, d2], dim=1)  # Concatenate along the channel dimension
        d2d = self.dec2(d2c)  # 128, 128, 128 -> 32, 128, 128

        d1d = self.dec1(d2d)  # 32, 128, 128 -> 1, 128, 128
        
        return d1d
    
# Specify a target name or fingerprint you want to deploy on
# target = "DPUCAHX8L_ISA0_SP"
target = "DPUCZDX8G_ISA1_B3136"
# Initialize inspector with target
inspector = Inspector(target)

# Start to inspect the float model
# Note: visualization of inspection results relies on the dot engine.If you don't install dot successfully, set 'image_format = None' when inspecting.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet5()
dummy_input = torch.randn(1, 1, 128, 1024)
inspector.inspect(model, (dummy_input,), device=device, output_dir="./inspect", image_format="png") 
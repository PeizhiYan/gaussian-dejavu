######################################
## Neural Network Code.              #
## Author: Peizhi Yan                #
##   Date: 03/27/2024                #
## Update: 03/27/2025                #
######################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PixelShuffleUpsample import PixelShuffleUpsample, Blur

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => ReLU) * 2
    Don't use batch normalization, because it will let the network ``know''
    which dataset the input images are probabily from.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            ##nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            ##nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(mid_channels, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, method='bilinear'):
        super().__init__()
        self.method = method

        # if bilinear, use the normal convolutions to reduce the number of channels
        if method == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        elif method == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.blur_layer = Blur()
            self.conv = DoubleConv(in_channels, out_channels)
        elif method == 'pixelshuffle':
            self.up = PixelShuffleUpsample(in_channels)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if self.method == 'transpose':
            x1 = self.blur_layer(x1)
        if x2 is not None:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class EDNet(nn.Module):
    # Encoder-Decoder Network
    def __init__(self, in_channels, out_channels, img_size=256, upmethod='pixelshuffle'):
        super(EDNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upmethod = upmethod
        self.img_size = img_size  # S = 320
        ac = 3 # 0 or 3

        # resize layers
        self.resize_input = T.Resize(img_size)   # resize layer for input image
        self.resize_0 = T.Resize(img_size // 16) # resize layer for UV position map
        self.resize_1 = T.Resize(img_size // 8)  # resize layer for UV position map
        self.resize_2 = T.Resize(img_size // 4)  # resize layer for UV position map
        self.resize_3 = T.Resize(img_size // 2)  # resize layer for UV position map

        # encoder layers
        self.inc = (DoubleConv(in_channels, 64))   # S    = 320    (input layer)
        self.down1 = (Down(64, 64))                # S/2  = 160
        self.down2 = (Down(64, 128))               # S/4  = 80
        self.down3 = (Down(128, 256))              # S/8  = 40
        self.down4 = (Down(256, 512))              # S/16 = 20

        # decoder layers
        self.up1 = (Up(512+ac, 256, upmethod))      # S/8  = 40
        self.up2 = (Up(256+ac, 128, upmethod))      # S/4  = 80
        self.up3 = (Up(128+ac, 64, upmethod))       # S/2  = 160
        self.up4 = (Up(64+ac, 64, upmethod))        # S    = 320
        self.outc = (OutConv(64, out_channels))     # S    = 320    (output layer)

    def get_encoder_parameters(self):
        # Collect parameters from encoder layers
        encoder_layers = [self.inc, self.down1, self.down2, self.down3, self.down4]
        return [param for layer in encoder_layers for param in layer.parameters()]

    def get_decoder_parameters(self):
        # Collect parameters from decoder layers
        decoder_layers = [self.up1, self.up2, self.up3, self.up4, self.outc]
        return [param for layer in decoder_layers for param in layer.parameters()]

    def encode(self, x):
        # x: [N,3,H,W]  input image
        # returns f:    feature map

        # resize the input image
        x = self.resize_input(x)    # [N,3,S,S]

        # encoding
        f = self.inc(x)      # S
        f = self.down1(f)    # S/2
        f = self.down2(f)    # S/4
        f = self.down3(f)    # S/8
        f = self.down4(f)    # S/16

        return f

    def decode(self, f, p):
        # x: [N,3,H,W]  input image
        # p: [N,3,S,S]  initial uv positions
        # returns y: output uv offsets

        # resize uv position maps
        p0 = self.resize_0(p) # S/16
        p1 = self.resize_1(p) # S/8
        p2 = self.resize_2(p) # S/4
        p3 = self.resize_3(p) # S/2
        
        # decoding
        #"""
        f = self.up1(torch.cat([f, p0], dim=1))   # S/8
        f = self.up2(torch.cat([f, p1], dim=1))   # S/4
        f = self.up3(torch.cat([f, p2], dim=1))   # S/2
        f = self.up4(torch.cat([f, p3], dim=1))   # S
        y = self.outc(f)                          # S
        #y = self.outc(torch.cat([f, p], dim=1))   # S
        """
        f = self.up1(f)    # S/8
        f = self.up2(f)    # S/4
        f = self.up3(f)    # S/2
        f = self.up4(f)    # S
        y = self.outc(f)   # S
        """
        
        return y

    def forward(self, x, p):
        # x: [N,3,H,W]  input image
        # p: [N,3,S,S]  initial uv positions
        f = self.encode(x)
        y = self.decode(f, p)
        return f, y




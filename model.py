import torch
import torch.nn as nn
import config
import os

# Set the environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class UNet(nn.Module):
    def __init__(self, in_channels=config.in_channels, out_channels=config.out_channels): # 4 classes: tumour core, enhancing tumor, edema, background
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 320)

        # Decoder (upsampling)
        self.dec5 = self.conv_block(320, 256)
        self.dec4 = self.conv_block(256 + 256, 128)
        self.dec3 = self.conv_block(128 + 128, 64)
        self.dec2 = self.conv_block(64 + 64, 32)
        self.dec1 = nn.Conv3d(32 + 32, out_channels, kernel_size=1)

        # Deep supervision outputs
        #self.out5 = nn.Conv3d(256, out_channels, kernel_size=1)
        #self.out4 = nn.Conv3d(128, out_channels, kernel_size=1)
        #self.out3 = nn.Conv3d(64, out_channels, kernel_size=1)
        #self.out2 = nn.Conv3d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.interpolate(enc1, scale_factor=0.5, mode='trilinear', align_corners=True))
        enc3 = self.enc3(nn.functional.interpolate(enc2, scale_factor=0.5, mode='trilinear', align_corners=True))
        enc4 = self.enc4(nn.functional.interpolate(enc3, scale_factor=0.5, mode='trilinear', align_corners=True))
        enc5 = self.enc5(nn.functional.interpolate(enc4, scale_factor=0.5, mode='trilinear', align_corners=True))

        # Decoder
        dec5 = self.dec5(enc5)

        # Print nvidia-smi for memory usage monitoring
        os.system('nvidia-smi')

        dec4 = self.dec4(torch.cat([nn.functional.interpolate(dec5, enc4.size()[2:], mode='trilinear', align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([nn.functional.interpolate(dec4, enc3.size()[2:], mode='trilinear', align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.functional.interpolate(dec3, enc2.size()[2:], mode='trilinear', align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.functional.interpolate(dec2, enc1.size()[2:], mode='trilinear', align_corners=True), enc1], dim=1))

        # Deep supervision outputs
        #out5 = self.out5(dec5)
        #out4 = self.out4(dec4)
        #out3 = self.out3(dec3)
        #out2 = self.out2(dec2)

        return dec1 # We simiply output the logits bc the loss function we will use (CrossEntropyLoss) likes to take the logits as input


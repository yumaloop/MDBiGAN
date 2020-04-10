import os
import numpy as np
import torch
import torch.nn as nn

def to_cuda(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)
    
class Flatten(nn.Module):
    def forward(self, x):
        x =  x.view(x.size(0), -1)
        # print(x.size())
        return x

class ImageAutoEncoder(nn.Module):
    """
    Z'm = E(X)
    Video Encoder: X → Z'm
    """
    def __init__(self, 
                 n_channel=3, 
                 dim_zm=2,
                 dim_zc=2):
        
        super(VideoAutoEncoder, self).__init__()
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zc + dim_zm
        
        self.encoder_zm = nn.Sequential(
            # Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(n_channel, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, 4, 1, 0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(6561, self.dim_zm),
            nn.Tanh()
        )
        self.encoder_zc = nn.Sequential(
            # Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(n_channel, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, 4, 1, 0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(6561, self.dim_zc),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.dim_z, 512, 6),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, n_channel, 4, 2, 1),
            nn.BatchNorm2d(n_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        x: 4D-Tensor (batch_size, channel, height, width)
        """
        zm = self.encoder_zm(x) # (batch_size, dim_zm)
        zc = self.encoder_zc(x) # (batch_size, dim_zc)
        z = torch.cat([zc, zm], dim=1) # z: (batch_size, dim_z)
        z = z.reshape(z.size(0), self.dim_z, 1, 1) # z: (batch_size, dim_z, height, width)
        x_recon = self.decoder(z)
        return x_recon

    def _forward(self, x):
        """
        x: 5D-Tensor (batch_size, channel, video_len, height, width)
        """
        x = x.permute(2, 1, 0, 3, 4) # x: (video_len, channel, batch_size. height, width)        
        zm = []
        for x_t in x:
            x_t = x_t.permute(1, 0, 2, 3) # x_t: (batch_size, channel, height, width)
            zm_t = self.encoder_zm(x_t)  # zm_t: (batch_size, dim_z_motion)
            zm.append(zm_t)
        zm = torch.stack(zm)
        zm = zm.permute(1, 0, 2)
        zm = to_cuda(zm) # zm: 3D-Tensor (batch_size, video_len, dim_z_motion)
        return zm



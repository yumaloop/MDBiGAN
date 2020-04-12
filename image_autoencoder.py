import os
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_cuda(x):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return x.to(device)
    
class Flatten(nn.Module):
    def forward(self, x):
        x =  x.view(x.size(0), -1)
        # print(x.size())
        return x

class ImageAutoEncoder(nn.Module):
    """
    Zm,Zc = E(X), X' = D(Zm,Zc)
    """
    def __init__(self, 
                 n_channel=3, 
                 dim_zm=2,
                 dim_zc=2):
        
        super(ImageAutoEncoder, self).__init__()
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zc + dim_zm
        
        self.encoder_zm = nn.Sequential(
            # Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(n_channel, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 4, 1, 0),
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
            nn.Conv2d(n_channel, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 4, 1, 0),
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
            nn.ConvTranspose2d(self.dim_z, 256, 6),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, n_channel, 4, 2, 1),
            nn.BatchNorm2d(n_channel),
            nn.Sigmoid()
            # nn.LeakyReLU(0.2, inplace=True)
        )
    def _encoder_zm(self, x):
        return self.encoder_zm(x)

    def _encoder_zc(self, x):
        return self.encoder_zc(x)

    def _decoder(self, zm, zc):
        z  = torch.cat([zm, zc], dim=1) # z: (batch_size, dim_z)
        z  = z.view(z.size(0), self.dim_z, 1, 1) # z: (batch_size, dim_z, height, width)
        return self.decoder(z)

    def _sample_zm(self, num):
        return torch.randn(num, self.dim_zm).to(device)

    def _sample_zc(self, num):
        return torch.randn(num, self.dim_zc).to(device)

    def forward(self, x):
        """
        x: 4D-Tensor (batch_size(=video_len), channel, height, width)
        """
        zm = self._encoder_zm(x) # zm: (batch_size, dim_zm)
        zc = self._encoder_zc(x) # zc: (batch_size, dim_zc)
        # zm_randn = self._sample_zm(x.shape[0]) # z_randn: (batch_size, dim_zm)
        x_recon_z  = self._decoder(zm, zc) # x_recon: (batch_size, channel, height, width)
        # x_recon_zc = self._decoder(zm_randn, zc)
        return x_recon_z, zm, zc

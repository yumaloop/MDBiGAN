import numpy as np
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * to_cuda(torch.FloatTensor(x.size()).normal_())
        return x

    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Generator(nn.Module):
    """
    X' = G(Zc, Zm)
    Video Generator: (Zc, Zm) → X'
    """
    def __init__(self, dim_zc=2, dim_zm=2, dim_e=4, n_channels=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(VideoGenerator, self).__init__()
        self.n_channels = n_channels
        self.dim_e = dim_e
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = self.dim_zc + self.dim_zm
        
        self.device = device
        
        # LSTM (Recurrent Network)
        self.num_layers = 3
         self.lstm = nn.LSTM(self.dim_e, 
                             self.dim_zm,
                             self.num_layers,
                             batch_first=True)
        # e  to zm
        self.zm_to_e = nn.Sequential(
            nn.Linear(self.dim_zm, self.dim_e)
            nn.BatchNorm1d(self.dim_e),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.generator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(self.dim_z, 512, 6, 0, 0),
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
            
            nn.ConvTranspose2d(64, self.n_channels, 4, 2, 1),
            nn.BatchNorm2d(self.n_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def _sample_zm(self, batch_size, video_len):
        """
        zm: (batch_size, video_len, dim_zm)
        """
        # input_e: (batch_size, video_len, dim_e)
        input_e = torch.randn(batch_size, video_len, self.dim_e) 
        
        # Initialize cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.dim_zm).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.dim_zm).to(self.device)
        
        # zm: (batch_size, video_len, dim_zm)
        zm, (hn, cn) = self.lstm(input_e, (h0, c0)) 
        return zm.to(self.device)

    def _sample_zc(self, batch_size, video_len):
        """
        zc: (batch_size, video_len, dim_zc)
        """
        zc = torch.randn(batch_size, 1, self.dim_zc).repeat(1, video_len, 1)
        return zc.to(device)

    def _sample_z(self, batch_size, video_len):
        """
        z: (batch_size, video_len, dim_z)
        """
        zm = _sample_zm(self, batch_size, video_len) # zm: (batch_size, video_len, dim_zm)
        zc = _sample_zc(self, batch_size, video_len) # zc: (batch_size, video_len, dim_zc)
        z = torch.cat([zc, zm], dim=2) # z: (batch_size, video_len, dim_z)
        return z
    
    def forward(self, batch_size, video_len):
        """
        v_fake: (batch_size, video_len, channel, height, width)
        """
        z = _sample_z(batch_size, video_len) # z: (batch_size, video_len, dim_z)
        z = z.premute(1, 0, 2) # z: (video_len, batch_size, dim_z)
        
        # zt: (batch_size, dim_z)
        # x_fake: (batch_size, channel, height, width)
        v_fake = torch.Tensor([x_fake = self.generator(zt) for zt in z]) # x_fake: (video_len, batch_size, channel, height, width)
        v_fake = v_fake.permute(1, 0, 2, 3, 4) # v_fake: (batch_size, video_len, channel, height, width)
        return v_fake.to(self.device) 


class ImageDiscriminator(nn.Module):
    """
    {1, 0} = DI(X)
    Image Discriminator: {X[i], X'[i]} → {1, 0}
    """
    def __init__(self, dim_zc=2, dim_zm=2, dim_e=4, n_channels=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(ImageDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zm + dim_zc
        self.use_noise = use_noise

        self.image_discriminator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: 4D-Tensor (batch_size, channel, height, width)
        output: (batch_size, 1)
        """
        output = image_discriminator(x)
        return output.to(self.device)

class VideoDiscriminator(nn.Module):
    """
    {1, 0} = DI(X)
    Image Discriminator: {X[i], X'[i]} → {1, 0}
    """
    def __init__(self, dim_zc=2, dim_zm=2, dim_e=4, n_channels=3, 
                 use_noise=False, device='cuda:0', noise_sigma=None):
        
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zm + dim_zc
        self.use_noise = use_noise

        self.video_discriminator = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, 64, 4, 1, 0),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, 4, 1, 0),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, 4, 1, 0),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: 5D-Tensor (batch_size, video_len, channel, height, width)
        output: (batch_size, 1)
        """
        output = video_discriminator(x)
        return output.to(self.device)
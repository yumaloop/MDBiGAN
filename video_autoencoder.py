import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, x):
        x =  x.view(x.size(0), -1)
        # print(x.size())
        return x


class VideoAutoEncoder(nn.Module):
    def __init__(self, dim_zc=2, dim_zm=2, input_size=16, hidden_size=16, device='cuda:0', num_layers=3, n_channel=3):
        super(VideoAutoEncoder, self).__init__()

        self.dim_zc = dim_zc
        self.dim_zm = dim_zm
        self.dim_z = dim_zc + dim_zm

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        
        self.conv = nn.Sequential(
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
            
            nn.Conv2d(64, 1, 4, 1, 0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(7056, self.input_size)
        )
        
        self.encoder_zm = nn.Sequential(
            nn.Linear(self.input_size, self.dim_zm),
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
        """
        x: 5D-Tensor (batch_size=1, video_len, channel, height, width)
        """
        # Initialize hidden state with zeros
        # x: (batch_size=1, video_len, channel, height, width)
        B, S, ch, h, w = x.shape
        
        input_vec = self.conv(torch.squeeze(x)) # input_vec: (video_len, input_size)
        input_vec = torch.unsqueeze(input_vec, 0) # input_vec: (batch_size=1, video_len, input_size)
        
        # Initialize cell state
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)
        
        output_vec, (hn, cn) = self.lstm(input_vec, (h0, c0)) # output_vec: (batch_size=1, video_len, input_size)
        zm = self.encoder_zm(torch.squeeze(output_vec)) # zm: (video_len, dim_zm)
        return zm

    def _encoder_zc(self, x):
        return self.encoder_zc(torch.squeeze(x))

    def _decoder(self, zm, zc):
        z  = torch.cat([zm, zc], dim=1) # z: (batch_size, dim_z)
        z  = z.view(z.size(0), self.dim_z, 1, 1) # z: (batch_size, dim_z, height, width)
        x_hat =  torch.unsqueeze(self.decoder(z), 0) # x_hat: (batch_size, video_len channel, height, width)
        return x_hat

    def forward(self, x):
        """
        x: 5D-Tensor (batch_size=1, video_len, channel, height, width)
        """
        zm = self._encoder_zm(x) # zm: (video_len, dim_zm)
        zc = self._encoder_zc(x) # zc: (video_len, dim_zc)
        x_hat = self._decoder(zm, zc) # x_hat: (batch_size, video_len channel, height, width)
        return x_hat, zm, zc

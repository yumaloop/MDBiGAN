import os
import numpy as np
import torch
from torch import nn
# opt.spectral_normm
from torch.utils.data import Dataset, DataLoader, TensorDataset

def to_cuda(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)

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

class ImageEncoder(nn.Module):
    """
    Z'm = E(X)
    Image Encoder: X → Z'm
    """
    def __init__(self, 
                 n_channel=3, 
                 dim_z_motion=16):
        
        super(ImageEncoder, self).__init__()

        self.infer_image = nn.Sequential(
            """
            """
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
            
            nn.Conv2d(64, self.dim_z_motion, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(100, dim_z_motion),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: 4D-Tensor (batch_size, channel, height, width)
        """
        return self.infer_image(x) # (batch_size, dim_z_motion)

    def _forward(self, x):
        """
        x: 5D-Tensor (batch_size, channel, video_len, height, width)
        """
        x = x.permute(2, 1, 0, 3, 4) # x: (video_len, channel, batch_size. height, width)        
        zm = []
        for x_t in x:
            x_t = x_t.permute(1, 0, 2, 3) # x_t: (batch_size, channel, height, width)
            zm_t = self.infer_image(x_t)  # zm_t: (batch_size, dim_z_motion)
            zm.append(zm_t)
        zm = torch.stack(zm)
        zm = zm.permute(1, 0, 2)
        zm = to_cuda(zm) # zm: 3D-Tensor (batch_size, video_len, dim_z_motion)
        return zm



class VideoGenerator(nn.Module):
    """
    X' = G(Zc, Zm)
    Video Generator: (Zc, Zm) → X'
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content, 
                 dim_z_motion,
                 video_length=20, 
                 ngf=32,
                 use_noise=False, 
                 noise_sigma=None):
        
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.dim_e = dim_z_motion
        self.video_length = video_length
        
        # GRU
        self.recurrent = nn.GRUCell(self.dim_e, self.dim_z_motion)
        self.gru_linear = nn.Linear(self.dim_z_motion, self.dim_e)
        self.gru_bn = nn.BatchNorm1d(self.dim_z_motion)

        self.main = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
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
            
            nn.ConvTranspose2d(64, self.n_channels, 4, 2, 1),
            nn.BatchNorm2d(self.n_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def sample_z_motion(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        
        e_t = self.get_iteration_noise(num_samples)
        h_t = self.get_gru_initial_state(num_samples)
        
        outputs=[]
        for i in range(video_len):            
            h_t = self.recurrent(e_t, h_t)
            e_t = self.gru_linear(h_t)
            outputs.append(h_t)
            
        outputs = [self.gru_bn(h_t) for h_t in outputs]
        outputs = torch.stack(outputs) # outputs: (video_len, num_samples, dim_z_motion)
        outputs = outputs.permute(1, 0, 2) # outputs: (num_samples, video_len, dim_z_motion)
        z_m = to_cuda(outputs) 
        return z_m

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32) # (10,1)
        content = np.tile(content, (video_len, 1, 1)) # (20, 10, 1)
        content = np.transpose(content, (1, 0, 2))                
        z_c = torch.FloatTensor(content)
        z_c = to_cuda(z_c) # z_c: (num_samples, video_len, dim_z_content)
        return z_c

    def sample_z(self, num_samples, video_len=None):
        z_c = self.sample_z_content(num_samples, video_len) # (batch_size, video_len, dim_z_content)
        z_m = self.sample_z_motion(num_samples, video_len) # (batch_size, video_len, dim_z_motion)
        z = torch.cat((z_c, z_m), dim=2)
        z = to_cuda(z)
        return z, z_c, z_m
    
    def sample_videos(self, num_samples, z_motion=None, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        z, z_c, z_m = self.sample_z(num_samples, video_len)

        if z_motion is not None:
            z_m = z_motion

        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        h = to_cuda(h)
        return h, z, z_c, z_m

    def sample_images(self, num_samples):
        z, z_c, z_m = self.sample_z(num_samples, video_len=1)
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = to_cuda(h)
        z   = torch.squeeze(z)
        z_c = torch.squeeze(z_c)
        z_m = torch.squeeze(z_m)
        return h, z, z_c, z_m
    
    def sample_images_and_videos(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        z, z_c, z_m = self.sample_z(num_samples)  
        
        # fake images
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        images_fake = to_cuda(h)
        
        # fake videos
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        videos_fake = to_cuda(h)

        return images_fake, videos_fake, z, z_c, z_m

    def get_gru_initial_state(self, num_samples):
        # Random values following standard gaussi
        return to_cuda(torch.zeros(num_samples, self.dim_z_motion))

    def get_iteration_noise(self, num_samples):
        # Random values following standard gauss
        return to_cuda(torch.FloatTensor(num_samples, self.dim_e).normal_())



class ImageDiscriminator(nn.Module):
    """
    {1, 0} = DI(X)
    Image Discriminator: {X[i], X'[i]} → {1, 0}
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content,
                 dim_z_motion,
                 dropout,                 
                 ndf=16, 
                 video_length=20, 
                 use_noise=False, 
                 noise_sigma=None):
        
        super(ImageDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.dropout = dropout
        self.video_length = video_length
        self.use_noise = use_noise

        self.infer_x = nn.Sequential(
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
        
        self.infer_zm = nn.Sequential(
            nn.Linear(self.dim_z_motion, 128)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Dropout(p=self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, zm):
        """
        x: 4D-Tensor (num_images, channel, height, width)
        zm: 2D-Tensor (num_images, dim_z_motion)
        """
        output_x = self.infer_x(x) # (num_images, 1)
        output_zm = self.infer_zm(zm) # (num_images, 1)
        output = torch.cat([output_x, output_zm], dim=1) # (num_images, 2)
        output = output.squeeze()
        output = to_cuda(output)
        return output
    

class VideoDiscriminator(nn.Module):
    """
    {1, 0} = DV(X)
    Video Discriminator: {X, X'} → {1, 0}
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content,
                 dim_z_motion,
                 dropout,
                 video_length=20,
                 n_output_neurons=1, 
                 bn_use_gamma=True, 
                 use_noise=False, 
                 noise_sigma=None, 
                 ndf=16):
        
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.dropout = dropout
        self.n_output_neurons = n_output_neurons
        self.video_length = video_length
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        # input_shape : (batch_size, ch_in, duration, height, width)
        self.infer_x = nn.Sequential(
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
        
        self.infer_zm = nn.Sequential(
            nn.Linear(self.dim_z_motion * self.video_length, 128)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Dropout(p=self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, zm):
        """
        x: 5D-Tensor (batch_size, video_len, channel, height, widt h)
        zm: 3D-Tensor (batch_size, video_len, dim_z_motion)
        """
        zm = zm.contiguous().view(zm.size(0), int(zm.size(1)*zm.size(2))) # zm: (batch_size, dim_z_motion*video_length)
        
        output_x = self.infer_x(x) # (batch_size, 1)
        output_zm = self.infer_zm(zm) # (batch_size, 1)
        
        output = torch.cat([output_x, output_zm], dim=1) # (batch_size, 2)
        output = output.squeeze()
        output = to_cuda(output)
        return output

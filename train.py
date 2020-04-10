import os
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchsummary import summary
from tqdm import tqdm
from utils import make_save_image
from dataset import CustomMovingMNIST
from models import VideoGenerator, ImageDiscriminator, VideoDiscriminator, ImageEncoder, to_cuda, weights_init_normal, Noise

# Clear GPU cache
torch.cuda.empty_cache()

# Hyper params
num_epochs = 1000
batch_size = 16
num_samples = batch_size
num_videos = batch_size
num_images = batch_size * 3
video_len = 20
log_interval=10
dim_zm = 16
dim_zc = 64


# Load dataset
dataset = CustomMovingMNIST(transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_num  = dataset.__len__()
batch_num = data_loader.__len__()

# Generator, Discriminator, Encoder
netE  = to_cuda(ImageEncoder(n_channels=1, dim_z_motion=dim_zm))
netG  = to_cuda(VideoGenerator(n_channels=1, dim_z_content=dim_zc, dim_z_motion=dim_zm))
netDI = to_cuda(ImageDiscriminator(n_channels=1, dim_z_content=dim_zc, dim_z_motion=dim_zm, dropout=0.1, use_noise=True, noise_sigma=0.2))
netDV = to_cuda(VideoDiscriminator(n_channels=1, dim_z_content=dim_zc, dim_z_motion=dim_zm, dropout=0.1, use_noise=True, noise_sigma=0.2))

# Summary
# print(summary(netE, (1, 20, 64, 64)))
# print(summary(netG, (1, 20, 64, 64)))

# Initialize model weights
netE.apply(weights_init_normal)
netG.apply(weights_init_normal)
netDI.apply(weights_init_normal)
netDV.apply(weights_init_normal)

# Optimizers
optim_GE = torch.optim.Adam(list(netE.parameters()) + list(netG.parameters()), lr=0.0002, betas=(0.5, 0.999))
optim_DI = torch.optim.Adam(netDI.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_DV = torch.optim.Adam(netDV.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function: Binary cross entropy 
criterion = torch.nn.BCELoss()

# losses
GE_losses_per_epoch=[]
DI_losses_per_epoch=[]
DV_losses_per_epoch=[]

start_time = time.time()

# training
for epoch in range(num_epochs):    
    
    GE_losses_per_batch=[]
    DI_losses_per_batch=[]
    DV_losses_per_batch=[]

    for batch_num, (videos_real, motion_label) in enumerate(data_loader):
        netE.train()
        netG.train()
        netDI.train()
        netDV.train()
        
        optim_GE.zero_grad()
        optim_DV.zero_grad()
        
        # videos_real (const.)
        videos_real = to_cuda(videos_real.type(torch.FloatTensor))        

        # zm_fake (netE)
        zm_fake = netE(videos_real)

        # -----------------------------
        # Train Discriminator (Video)
        # -----------------------------
        optim_DV.zero_grad()

        # images_fake, zc_real, zm_real (netG)
        videos_fake, z_real, zc_real, zm_real = netG.sample_videos(num_videos)

        # target label
        if epoch >= 20:
            target_images_real = to_cuda(torch.ones(num_videos * video_len))
            target_images_fake = to_cuda(torch.zeros(num_videos * video_len))
            target_videos_real = to_cuda(torch.ones(num_videos))
            target_videos_fake = to_cuda(torch.zeros(num_videos))
        else:
            target_images_real = to_cuda(torch.ones(num_videos * video_len)  - torch.randn(num_videos * video_len) * 0.25)
            target_images_fake = to_cuda(torch.zeros(num_videos * video_len) + torch.randn(num_videos * video_len) * 0.25)
            target_videos_real = to_cuda(torch.ones(num_videos)  - torch.randn(num_videos) * 0.25)
            target_videos_fake = to_cuda(torch.zeros(num_videos) + torch.randn(num_videos) * 0.25)
        
        d_videos_real = netDV(videos_real, zm_fake.detach())
        d_videos_fake = netDV(videos_fake.detach(), zm_real.detach())
        
        DV_real_loss = criterion(d_videos_real, target_videos_real)
        DV_fake_loss = criterion(d_videos_fake, target_videos_fake)
        DV_loss = (DV_real_loss + DV_fake_loss) / 2
        
        DV_loss.backward(retain_graph=True)
        optim_DV.step()
        
        # -----------------------------
        # Train Discriminator (Image)
        # -----------------------------
        optim_DI.zero_grad()

        random_index = torch.randperm(num_images)

        # images_real (const.)
        images_real = videos_real.permute(0, 2, 1, 3, 4)
        images_real = images_real.view(int(images_real.size(0)*images_real.size(1)), images_real.size(2), images_real.size(3), images_real.size(4))
        images_real = images_real[random_index]
        images_real = images_real[0:num_images]
        
        # images_fake, zc_real, zm_real (netG)
        images_fake, z_real, zc_real, zm_real = netG.sample_images(num_images)

        # zm_fake (netE)
        zm_fake = zm_fake.contiguous().view(int(zm_fake.size(0)*zm_fake.size(1)), zm_fake.size(2))
        zm_fake = zm_fake[random_index]
        zm_fake = zm_fake[0:num_images]

        # target label
        if epoch >= 20:
            target_images_real = to_cuda(torch.ones(num_images))
            target_images_fake = to_cuda(torch.zeros(num_images))
            target_videos_real = to_cuda(torch.ones(num_samples))
            target_videos_fake = to_cuda(torch.zeros(num_samples))
        else:
            target_images_real = to_cuda(torch.ones(num_images)  - torch.randn(num_images) * 0.25)
            target_images_fake = to_cuda(torch.zeros(num_images) + torch.randn(num_images) * 0.25)
            target_videos_real = to_cuda(torch.ones(num_samples)  - torch.randn(num_samples) * 0.25)
            target_videos_fake = to_cuda(torch.zeros(num_samples) + torch.randn(num_samples) * 0.25)
        
        d_images_real = netDI(images_real, zm_fake.detach())
        d_images_fake = netDI(images_fake.detach(), zm_real.detach())
        
        DI_real_loss = criterion(d_images_real, target_images_real)
        DI_fake_loss = criterion(d_images_fake, target_images_fake)
        DI_loss = (DI_real_loss + DI_fake_loss) / 2
        
        DI_loss.backward(retain_graph=True)
        optim_DI.step()
        
        # -----------------------------
        # Train Generator and Encoder
        # -----------------------------    
        optim_GE.zero_grad()

        # target label (const.)
        target_images_real = to_cuda(torch.ones(num_images))
        target_images_fake = to_cuda(torch.zeros(num_images))
        target_videos_real = to_cuda(torch.ones(num_videos))
        target_videos_fake = to_cuda(torch.zeros(num_videos))

            # ---------------------------
            # Generator (videos)
            # ---------------------------
        
        # images_fake, videos_fake, zc_real, zm_real (netG)
        images_fake, videos_fake, z_real, zc_real, zm_real = netG.sample_images_and_videos(num_samples)
        # zm_fake (netE)
        zm_fake = netE(videos_real)

        d_videos_real = netDV(videos_real, zm_fake)
        d_videos_fake = netDV(videos_fake, zm_real)

        GV_loss_real = criterion(d_videos_real, target_videos_fake)
        GV_loss_fake = criterion(d_videos_fake, target_videos_real)
        GV_loss = (GV_loss_real + GV_loss_fake) / 2 

            # ---------------------------
            # Generator (images)
            # ---------------------------

        # images_real (const.)
        images_real = videos_real.permute(0, 2, 1, 3, 4)
        images_real = images_real.view(int(images_real.size(0)*images_real.size(1)), images_real.size(2), images_real.size(3), images_real.size(4))

        random_index = torch.randperm(num_images)
        
        zm_fake = zm_fake.contiguous().view(int(zm_fake.size(0)*zm_fake.size(1)), zm_fake.size(2))
        zm_real = zm_real.contiguous().view(int(zm_real.size(0)*zm_real.size(1)), zm_real.size(2))
        zm_fake = zm_fake[random_index]
        zm_real = zm_real[random_index]
        zm_fake = zm_fake[0:num_images]
        zm_real = zm_real[0:num_images]

        images_real = images_real[random_index]
        images_real = images_real[0:num_images]
        images_fake = images_fake[random_index]
        images_fake = images_fake[0:num_images]

        # images_real, images_fake: (num_images, ch, h, w) 
        # zm_real, zm_fake: (num_imagse, dim_z_motion)
        d_images_real = netDI(images_real, zm_fake)
        d_images_fake = netDI(images_fake, zm_real)
        
        GI_loss_real = criterion(d_images_real, target_images_fake)
        GI_loss_fake = criterion(d_images_fake, target_images_real)
        GI_loss = (GI_loss_real + GI_loss_fake) / 2
        
        G_loss = (GI_loss + GV_loss) / 2
        
        G_loss.backward(retain_graph=False)
        optim_GE.step()

        # -----------------------------
        # Logger messages
        # -----------------------------
                
        #  Batch-wise Loss
        GE_losses_per_batch.append(G_loss.item())
        DI_losses_per_batch.append(DI_loss.item())
        DV_losses_per_batch.append(DV_loss.item())
        
        if batch_num % log_interval == 0 and batch_num != 0:
            GE_loss_mean = sum(GE_losses_per_batch[-log_interval:]) / log_interval
            DI_loss_mean = sum(DI_losses_per_batch[-log_interval:]) / log_interval
            DV_loss_mean = sum(DV_losses_per_batch[-log_interval:]) / log_interval

            end_time = time.time()
            elapsed_time = end_time - start_time

            print("Epoch: {:>3}/{:>3} - Batch: {:>4}/{:>4} - GE_loss: {:<2.4f}, DI_loss: {:<2.4f}, DV_loss: {:<2.4f}, Time: {:<4.4f} (s)".format( \
                epoch, num_epochs, batch_num, int(data_num / batch_size), GE_loss_mean, DI_loss_mean, DV_loss_mean, elapsed_time))

            start_time = time.time()
            
        if batch_num % (log_interval*40) == 0 and batch_num != 0:
            # Generate images(videos)
            zm_fake = netE(videos_real)
            videos_fake, _, _, _ = netG.sample_videos(num_samples=5, z_motion=zm_fake)

            png_image_tensor = torch.Tensor(make_save_image(videos_fake, videos_real))
            save_image(png_image_tensor, os.path.join('./gen_images/epoch{}-batch{}_fakevideos.png'.format(epoch, batch_num)))
            
    # Epoch-wise Loss
    GE_losses_per_epoch.append(sum(GE_losses_per_batch) / len(GE_losses_per_batch))
    DI_losses_per_epoch.append(sum(DI_losses_per_batch) / len(DI_losses_per_batch))
    DV_losses_per_epoch.append(sum(DV_losses_per_batch) / len(DV_losses_per_batch))
    
    print("Epoch: {:>3}/{:>3} - GE_loss: {:<2.4f}, DI_loss: {:<2.4f}, DV_loss: {:<2.4f}".format( \
        epoch, num_epochs, GE_losses_per_epoch[epoch], DI_losses_per_epoch[epoch], DV_losses_per_epoch[epoch]))

    if epoch % 10 == 0:
        torch.save(netE.state_dict(),  'netE_'+str(epoch)+'.pt')
        torch.save(netG.state_dict(),  'netG_'+str(epoch)+'.pt')
        torch.save(netDI.state_dict(), 'netDI_'+str(epoch)+'.pt')
        torch.save(netDV.state_dict(), 'netDV_'+str(epoch)+'.pt')
    

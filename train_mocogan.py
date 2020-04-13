import numpy as np
import torch
import torchvision

from dataset import WeizmannHumanActionVideo
from mocogan import Generator, ImageDiscriminator, VideoDiscriminator, weights_init_normal

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

trans_data = torchvision.transforms.ToTensor()
trans_label = None
dataset = WeizmannHumanActionVideo(trans_data=None, trans_label=trans_label, train=True)

# train-test split
train_size = int(1.0 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print("train: ", len(train_dataset))
print("test: ", len(test_dataset))

# data_loader
batch_size=1
n_epochs=100


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=4)

"""
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, 
                                           num_workers=1)
"""

# model
netG = Generator(n_channel=3, dim_zm=2, dim_zc=2).to(device)
netDI = ImageDiscriminator(n_channel=3, dim_zm=2, dim_zc=2).to(device)
netDV = VideoDiscriminator(n_channel=3, dim_zm=2, dim_zc=2).to(device

# Initialize model weights
netG.apply(weights_init_normal)
netDI.apply(weights_init_normal)
netDV.apply(weights_init_normal)

# Optimizers
optim_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_DI = torch.optim.Adam(netDI.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_DV = torch.optim.Adam(netDV.parameters(), lr=0.0002, betas=(0.5, 0.999))


# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss(reduction='mean')
                                                               
real_label = 1
fake_label = 0

def S_1(video):
    """
    video: torch.Tensor()
        (batch_size, video_len, channel, height, width)
    image: torch.Tensor()
        (batch_size, channel, height, width)
    """
    idx = int(np.random.rand() * video.shape[1])
    image =  torch.squeeze(video[:, idx, :, :, :])
    return image

def S_T(video, T):
    """
    video: torch.Tensor()
        (batch_size, video_len, channel, height, width)
    """
    idx = int(np.random.rand() * (video.shape[1] - T))
    return video

def train_model(epoch):
    netG.train()
    netG.to(device)
    netDI.train()
    netDI.to(device)
    netDV.train()
    netDV.to(device)
    
    train_loss_G = 0
    train_loss_DI = 0

    for batch_idx, (batch_data, _) in enumerate(train_loader):
        # print(torch.cuda.memory_allocated(device))
        # x, batch_data: 5D-Tensor (batch_size=1, video_len, channel, height, width)
        x = batch_data.to(device) 

        # =====================================
        # (1) Update DI, DV network: 
        #     maximize   log ( DI ( SI(x) ) ) + log(1 - DI ( SI ( G(z) ) ) )
        #              + log ( DV ( SV(x) ) ) + log(1 - DV ( SV ( G(z) ) ) )
        # =====================================
        
        ## Train with all-real batch
        netDI.zero_grad()
        netDV.zero_grad()
        
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # Forward pass real batch through D
        output = netDI(real_cpu)

        # Calculate loss on all-real batch
        loss_DI_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_x = output.mean().item()

        ## Train with all-fake batch
        
        # Generate fake image batch with G
        v_fake = netG(noise)

        # x_hat: (video_len, channel, height, width)
        # zc: (video_len, dim_zc)        
        x_hat_z, zm, zc = model(x) 

        ## Train with all-fake batch


        BCE_loss_z  = criterion(x_hat_z,  x)
        # ZM_loss = torch.norm(1. - zm.std(dim=0), 2) + 1e-7
        ZC_loss = torch.norm(zc.std(dim=0), 2) + 1e-7
        # loss = MSE_loss_z + 0.5 * MSE_loss_zc + 0.5 * ZC_loss
        loss = BCE_loss_z + 0.005 * ZC_loss
        loss.backward() # compute accumulated gradients

        train_loss += loss.item()

        optimizer.step()

        print("epoch : {}/{}, batch : {}/{}, loss = {:.4f}, BCE_loss(Z) = {:.4f}, ZM_loss = {:.4f}, ZC_loss = {:.4f}".format(
            epoch + 1, n_epochs, batch_idx, int(len(train_dataset)/batch_size), loss.item(), BCE_loss_z.item(), 0.0, ZC_loss.item()))   
        
        del zc
        del zm
        del x_hat_z
        # del x_hat_zc
        del loss

    print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, n_epochs, train_loss / len(train_loader)))

                                                          
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
    

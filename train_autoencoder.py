import os
import glob
import cv2

import torch
import torchvision

from dataset import WeizmannHumanActionVideo
from image_autoencoder import ImageAutoEncoder


# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


"""
trans_data = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""

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
n_epochs=10


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
model = ImageAutoEncoder(n_channel=3, dim_zm=2, dim_zc=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

criterion = torch.nn.MSELoss()


# training
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    
    for batch_id, (batch_data, _) in enumerate(train_loader):
        # print(torch.cuda.memory_allocated(device))
        
        # batch_data: 5D-Tensor (batch_size=1, video_len, channel, height, width)
        # x: 4D-Tensor (video_len, channel, height, width)
        x = torch.squeeze(batch_data, dim=0).to(device) 

        if x.shape[0] >= 90:
            x = x[:77]

        print("x.shape: ", x.shape)
               
        optimizer.zero_grad()
    
        # x_hat: (video_len, channel, height, width)
        # zc: (video_len, dim_zc)        
        
        x_hat_z, x_hat_zc, zc = model(x) 
        # x_hat = model(x) 

        MSE_loss_z  = criterion(x_hat_z,  x)
        MSE_loss_zc = criterion(x_hat_zc, x)
        ZC_loss  = torch.norm(zc.std(dim=0), 2) + 1e-7
        loss = MSE_loss_z + MSE_loss_zc + 100 * ZC_loss
        loss.backward() # compute accumulated gradients
        
        train_loss += loss.item()

        optimizer.step()
                
        print("epoch : {}/{}, batch : {}/{}, loss = {:.4f}, MSE_loss(Z) = {:.4f}, MSE_loss(Zc) = {:.4f}, ZC_loss = {:.4f}".format(
            epoch + 1, n_epochs, batch_id, int(len(train_dataset)/batch_size), loss.item(), MSE_loss_z.item(), MSE_loss_zc.item(), ZC_loss.item()))   
        
        del zc
        del x_hat_z
        del x_hat_zc
        del loss
        
    print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, n_epochs, train_loss / len(train_loader)))


model_path = './trained_models/image_autoencoder.pth'
torch.save(model.to('cpu').state_dict(), model_path)

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
n_epochs=30


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

# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss(reduction='mean')

def train_model(epoch):
    model.train()
    model.to(device)
    train_loss = 0
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        # print(torch.cuda.memory_allocated(device))
        # batch_data: 5D-Tensor (batch_size=1, video_len, channel, height, width)
        
        x = torch.squeeze(batch_data, dim=0).to(device)
        # x: 4D-Tensor (video_len, channel, height, width)
        # print("x.shape: ", x.shape)

        optimizer.zero_grad()

        # x_hat: (video_len, channel, height, width)
        # zc: (video_len, dim_zc)        
        x_hat_z, zm, zc = model(x) 


        BCE_loss_z  = criterion(x_hat_z,  x)
        # MSE_loss_zc = criterion(x_hat_zc, x)
        ZM_loss = torch.norm(1. - zm.std(dim=0), 2) + 1e-7
        ZC_loss = torch.norm(zc.std(dim=0), 2) + 1e-7
        # loss = MSE_loss_z + 0.5 * MSE_loss_zc + 0.5 * ZC_loss
        loss = BCE_loss_z + 0.8 * ZM_loss + 0.5 * ZC_loss
        loss.backward() # compute accumulated gradients

        train_loss += loss.item()

        optimizer.step()

        print("epoch : {}/{}, batch : {}/{}, loss = {:.4f}, BCE_loss(Z) = {:.4f}, ZM_loss = {:.4f}, ZC_loss = {:.4f}".format(
            epoch + 1, n_epochs, batch_idx, int(len(train_dataset)/batch_size), loss.item(), BCE_loss_z.item(), ZM_loss.item(), ZC_loss.item()))   
        
        """
        print("epoch : {}/{}, batch : {}/{}, loss = {:.4f}, MSE_loss(Z) = {:.4f}, MSE_loss(Zc) = {:.4f}, ZC_loss = {:.4f}".format(
            epoch + 1, n_epochs, batch_idx, int(len(train_dataset)/batch_size), loss.item(), MSE_loss_z.item(), MSE_loss_zc.item(), ZC_loss.item()))   
        """

        del zc
        del zm
        del x_hat_z
        # del x_hat_zc
        del loss

    print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, n_epochs, train_loss / len(train_loader)))

def eval_model(epoch):
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(train_loader):

            # x: 4D-Tensor (video_len, channel, height, width)
            x = torch.squeeze(batch_data, dim=0).to(device)
            x_hat_z, zm, zc = model(x)
            
            del zm
            del zc

            if batch_idx == 0:
                n = x.size(0)
                comparison = torch.cat([x[:n], x_hat_z[:n]])
                # comparison = torch.cat([x[:n], x_hat_z.view(x.shape[0], 3, 96, 96)[:n], x_hat_zc.view(x.shape[0], 3, 96, 96)[:n]])
                torchvision.utils.save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                break

            return

if __name__ == "__main__":
    for epoch in range(n_epochs):
        train_model(epoch)
        eval_model(epoch)
        model_path = './trained_models/image_autoencoder'+str(epoch)+'.pth'
        torch.save(model.to('cpu').state_dict(), model_path)

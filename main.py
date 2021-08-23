import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim

from image_dataset import ImageDataset
from utils import show_batch
from gan import Discriminator, Generator


device = "cuda" if torch.cuda.is_available() else "cpu"
# Create dataset
filepath = '/home/weich/Documents/phd/data/'
files = os.path.join(filepath, 'synthetic_data')
image_dir = glob.glob(os.path.join(files, 'img*.png'))
mask_dir = glob.glob(os.path.join(files, 'mask*.png'))
images = [plt.imread(img) for img in image_dir]
images = [cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) for img in images]
masks = [plt.imread(mask) for mask in mask_dir]

dataset = ImageDataset(filepath=filepath)

# Split dataset
train_ratio = 0.7
shuffle = True
indices = list(range(len(dataset)))
split = int(np.floor(train_ratio * len(dataset)))
if shuffle:
    np.random.seed(1)
    np.random.shuffle(indices)

train_indices, test_indices = indices[:split], indices[split:]
val_indices, test_indices = test_indices[:len(test_indices)//2], test_indices[len(test_indices)//2:]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataloader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
val_dataloader = DataLoader(dataset, batch_size=4, sampler=val_sampler)
test_dataloader = DataLoader(dataset, batch_size=4, sampler=test_sampler)


#show_batch(test_dataloader)

# Train model
netG = Generator().to(device)
netD = Discriminator().to(device)

# Optimizers
lr = 0.0001
optim_G = optim.Adam(netG.parameters(), lr=lr)
optim_D = optim.Adam(netD.parameters(), lr=lr)
# Adversarial loss
loss = nn.BCELoss()

D_loss_list = []
G_loss_list = []
n_epochs = 10

for epoch in range(1, n_epochs + 1):

    print(f'Epoch {epoch}/{n_epochs}')
    print('-' * 10)
    epoch_loss = 0.0

    for imgs, masks in train_dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        masks = torch.cat([imgs, masks], dim=1)  # concatenate image and mask

        # Discriminator training
        optim_D.zero_grad()
        r_target = torch.ones(len(masks), dtype=torch.float, device=device)
        output = netD(masks)
        # print(masks.shape, output.shape, r_target.shape)
        r_loss = loss(output, r_target)
        r_loss.backward()

        g_masks = netG(imgs)
        g_masks = torch.cat([imgs, g_masks], dim=1)
        f_target = torch.zeros(len(g_masks), dtype=torch.float, device=device)
        output = netD(g_masks.detach())
        f_loss = loss(output, f_target)
        f_loss.backward()

        D_loss = r_loss + f_loss
        D_loss_list.append(D_loss)
        optim_D.step()

        # Generator training
        optim_G.zero_grad()
        g_output = netD(g_masks)
        G_loss = loss(g_output, r_target)
        G_loss.backward()
        optim_G.step()
        G_loss_list.append(G_loss)

    print(f'D_loss: {D_loss:.4f}, G_loss: {G_loss:.4f}')
import matplotlib.image as mpimg
import cv2
import os
import glob

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.data_dir = os.path.join(filepath, 'synthetic_data')
        self.imgs = glob.glob(os.path.join(self.data_dir, 'img*.png'))
        self.masks = glob.glob(os.path.join(self.data_dir, 'mask*.png'))
        self.file_list = [os.path.basename(f).split('.')[0] for f in self.imgs]
        self.img_dim = (128, 96)
        self.transform = transform

    def __getitem__(self, idx):
        img = mpimg.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, self.img_dim)
        img = torch.tensor(img)
        mask = mpimg.imread(self.masks[idx])
        mask = cv2.resize(mask, self.img_dim)
        mask = torch.tensor(mask, dtype=torch.float)  # torch.long
        # Permute axis (n, c, w, h)
        img = img.permute(2, 0, 1)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.file_list)

    def get_filename(self, idx):
        return self.file_list[idx]
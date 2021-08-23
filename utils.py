import matplotlib.pyplot as plt
import torch


def show_batch(dataloader):
    n = dataloader.batch_size
    imgs, masks = next(iter(dataloader))
    print(len(imgs), imgs.shape)
    imgs = imgs.permute(0, 2, 3, 1)
    plt.figure(figsize=(12, 5))
    for i in range(dataloader.batch_size):
        ax1 = plt.subplot(2, n, i+1)
        ax1.imshow(imgs[i])
        ax1.axis('off')
        ax2 = plt.subplot(2, n, n+i+1)
        ax2.imshow(masks[i])
        ax2.axis('off')
    plt.show()
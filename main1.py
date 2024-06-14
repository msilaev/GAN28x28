import itertools

import numpy as np
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import time
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

channels = 1
img_size = 28
#img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
latent_dim = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Hout = (Hin - 1)×stride − 2×padding + kernel_size

        self.main = nn.Sequential(
            # Hout = (Hin - 1)×stride − 2×padding + kernel_size
            # dim, channels, kernel, stride, padding
            # Hin = 1, stride = 1, padding = 0,  kernel_size = 4
            # 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 3*2 - 2*1 + 4 = 8
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 2*(7) - 2 + 4 = 16
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, channels,    2,      2,      2, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # Hout = (Hin - 1)×stride − 2×padding + kernel_size
                             # dim, channels, kernel, stride, padding
            # 15*2 - 2*2 + 2 = 28
            # state size. (ngf) x 28 x 28
            #nn.ConvTranspose2d(ngf, channels, 4, 1, 0, bias=False),
            nn.Tanh()
            # 31*1 + 4
            # state size. (channels) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(nc)x 28x28``
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 14x14``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 7x7 ``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DatasetMNIST(Dataset):  # inherit abstract class - 'Dataset'

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image ad ndarray type (H, W, C)
        # be carefull for converting dtype to np.uint8 (Unsigned integer (0 to 255))
        # in this example, We use ToTensor(), so we define the numpy array like (H, W, C)

        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
    latent_dim = 100  # suggested default. dimensionality of the latent space

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if torch.cuda.is_available() and ngpu > 0:
        print("CUDA!")
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    adversarial_loss = torch.nn.BCELoss()

    generator = Generator().to(device)
    generator.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    train = pd.read_csv('data/train.csv')
    dataset = DatasetMNIST(file_path='data/train.csv',
                           transform=transforms.Compose( [ transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])] ) )

    batch_size = 64  # suggested default, size of the batches
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True)

    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    real_label = 1
    fake_label = 0

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    #########################################
    nrows = 5
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

    real_batch = next(iter(dataloader))[0]
    #print("batch shape", real_batch.shape)
    #print ("data loader shape", dataloader.shape)

    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()

    for nrow, ncol in itertools.product(range(ncols//2), range(nrows)):
        axes[ncol][nrow].imshow(real_batch.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow].axis('off')

        axes[ncol][nrow+2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow+1].axis('off')
    plt.savefig ("results/dataset_im_start.png")
    plt.show()


    #########################################
    n_epochs = 10
    iteration = 0

    d_loss_arr=[]
    g_loss_arr = []
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader,0):

            iteration += 1

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(real_label),
                             requires_grad=False)

            # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(imgs.size(0), 1).fill_(fake_label),
                            requires_grad=False)

            #######################################
            real_cpu = imgs.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #######################################

            # ------------
            # Train Discriminator
            # ------------

            discriminator.zero_grad()
            output = discriminator(real_cpu).view(-1)

            real_loss = adversarial_loss(output, label)
            real_loss.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            fake = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake.detach()).view(-1)

            fake_loss = adversarial_loss(output, label)
            fake_loss.backward()
            D_G_z1 = output.mean().item()

            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            optimizer_G.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)

            g_loss = adversarial_loss(output, label)
            g_loss.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            d_loss_arr.append(d_loss.item())
            g_loss_arr.append(g_loss.item())

            if ((iteration + 1) %100) == 0:

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, n_epochs, i, len(dataloader),
                             d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

                nrows = 5
                ncols = 5
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
                plt.suptitle('EPOCH : {} | BATCH(ITERATION) : {}'.format(epoch + 1, i + 1))

                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()

                for nrow, ncol in itertools.product( range(ncols), range(nrows)):

                    axes[ncol][nrow].imshow(fake.permute(0, 2, 3, 1)[ncol*nrow + nrow], cmap='gray')
                    axes[ncol][nrow].axis('off')
                plt.show(block = False)
                plt.pause(2)

                plt.close()
        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
        )

#########################################
nrows = 5
ncols = 4

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

real_batch = next(iter(dataloader))[0]
# print("batch shape", real_batch.shape)
# print ("data loader shape", dataloader.shape)

with torch.no_grad():
    fake = generator(fixed_noise).detach().cpu()

for nrow, ncol in itertools.product(range(ncols // 2), range(nrows)):
    axes[ncol][nrow].imshow(real_batch.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
    axes[ncol][nrow].axis('off')

    axes[ncol][nrow + 2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
    axes[ncol][nrow + 1].axis('off')

plt.savefig ("results/dataset_im_fin.png")

plt.show()

d_loss_arr = np.array(d_loss_arr)
g_loss_arr = np.array(g_loss_arr)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss_arr,label="G")
plt.plot(d_loss_arr,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

plt.savefig ("results/loss.png")
plt.show()
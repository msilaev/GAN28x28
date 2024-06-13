import numpy as np
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import time
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

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

        def block(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize:  # Default
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # inplace=True : modify the input directly. It can slightly decrease the memory usage.
            return layers  # return list of layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            # Asterisk('*') in front of block means unpacking list of layers - leave only values(layers) in list
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  # np.prod(1, 28, 28) == 1*28*28
            nn.Tanh()  # result : from -1 to 1
        )

    def forward(self, z):  # z == latent vector(random input vector)
        img = self.model(z)  # (64, 100) --(model)--> (64, 784)
        img = img.view(img.size(0), *img_shape)  # img.size(0) == N(Batch Size), (N, C, H, W) == default --> (64, 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),  # (28*28, 512)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # result : from 0 to 1
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # flatten -> from (64, 1, 28, 28) to (64, 1*28*28)
        validity = self.model(img_flat)  # Discriminate -> Real? or Fake? (64, 784) -> (64, 1)
        return validity


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    channels = 1  # suggested default : 1, number of image channels (gray scale)
    img_size = 28  # suggested default : 28, size of each image dimension
    img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))

    latent_dim = 100  # suggested default. dimensionality of the latent space

    cuda = True if torch.cuda.is_available() else False  # GPU Setting

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if (torch.cuda.is_available() and ngpu > 0):
        print("CUDA!")

    adversarial_loss = torch.nn.BCELoss()

    generator = Generator().to(device)
    generator.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    #if cuda:
    #    print("CUDA!")
    #    generator.cuda()
    #    discriminator.cuda()
    #    adversarial_loss.cuda()

    train = pd.read_csv('train.csv')

    for index in range(1, 6):  # N : 5 (Number of Image)
        temp_image = train.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        temp_label = train.iloc[index, 0]
        print('Shape of Image : ', temp_image.shape)
        print('label : ', temp_label)

    dataset = DatasetMNIST(file_path='train.csv',
                           transform=transforms.Compose(
                               [  # transforms.Resize(img_size), # Resize is only for PIL Image. Not for numpy array
                                   transforms.ToTensor(),  # ToTensor() : np.array (H, W, C) -> tensor (C, H, W)
                                   transforms.Normalize([0.5], [0.5])]
                           ))

    batch_size = 64  # suggested default, size of the batches
    dataloader = DataLoader(  # torch.utils.data.DataLoader
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # suggested default - beta parameters (decay of first order momentum of gradients)
    b1 = 0.5
    b2 = 0.999

    # suggested default - learning rate
    lr = 0.0002

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Visualize result

    n_epochs = 10  # suggested default = 200
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0),
                             requires_grad=False)

            # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0),
                            requires_grad=False)

            # Configure input

            real_imgs = imgs.type(Tensor)  # As mentioned, it is no longer necessary to wrap the tensor in a Variable.
            # real_imgs = Variable(imgs.type(Tensor)) # requires_grad=False, Default! It's same.

            # sample noise 'z' as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

            print("noise z", z.shape)
            gen_imgs = generator(z)

            # ------------
            # Train Discriminator
            # ------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs),
                                         valid)  # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),
                                         fake)  # We are learning the discriminator now. So have to use detach()

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()  # If didn't use detach() for gen_imgs, all weights of the generator will be calculated with backward().
            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            optimizer_G.zero_grad()



            # gen_imgs.shape == torch.Size([64, 1, 28, 28])
            # Loss measures generator's ability to fool the discriminator
            #g_loss = -adversarial_loss(discriminator(gen_imgs), fake)
            # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)

            g_loss.backward()
            optimizer_G.step()

            # ------------
            # Real Time Visualization (While Training)
            # ------------

            sample_z_in_train = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
            # z.shape == torch.Size([64, 100])
            sample_gen_imgs_in_train = generator(sample_z_in_train).detach().cpu()
            # gen_imgs.shape == torch.Size([64, 1, 28, 28])

            if ((i + 1) % 200) == 0:  # show while batch - 200/657, 400/657, 600/657
                nrow = 1
                ncols = 5
                fig, axes = plt.subplots(nrows=nrow, ncols=ncols, figsize=(8, 2))
                plt.suptitle('EPOCH : {} | BATCH(ITERATION) : {}'.format(epoch + 1, i + 1))
                for ncol in range(ncols):
                    axes[ncol].imshow(sample_gen_imgs_in_train.permute(0, 2, 3, 1)[ncol], cmap='gray')
                    axes[ncol].axis('off')
                plt.show(block = False)
                plt.pause(1)

                plt.close()
        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
        )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

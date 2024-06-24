import itertools

import numpy as np
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchviz import make_dot

import torch.nn as nn
import torch
import pandas as pd

import matplotlib.pyplot as plt

from models.gan import Generator, Discriminator
from dataset_batch import DatasetMNIST
from plot_results import plot_results

channels = 1
img_size = 28
#img_shape = (channels, img_size, img_size)
# (Channels, Image Size(H), Image Size(W))
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

if __name__ == '__main__':

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
    latent_dim = 100  # suggested default. dimensionality of the latent space

    batch_size = 64  # suggested default, size of the batches

    #--------
    # learning parameters
    #--------
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    #--------
    # real and fake labels
    #--------
    real_label = 1
    fake_label = 0

    #--------
    # find available device
    #--------
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if torch.cuda.is_available() and ngpu > 0:
        print("CUDA!")
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    adversarial_loss = torch.nn.BCELoss()

    #--------
    # define models
    #--------
    generator = Generator().to(device)
    generator.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    train = pd.read_csv('data/train.csv')
    dataset = DatasetMNIST(file_path='data/train.csv',
                           transform=transforms.Compose( [ transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])] ) )

    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    #--------
    # plot untrained generator output
    #--------
    real_batch = next(iter(dataloader))[0]
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    fname = "results_gan/dataset_im_start.png"
    plot_results(fname, real_batch, fake)

    # ------------
    # Save computational graph in folder \result
    # ------------
    #fake1 = generator(fixed_noise)
    #dot = make_dot(fake1, params=dict(generator.named_parameters()), show_attrs=False, show_saved=False)
    #dot.render("computation_graph_dicrim", format="png")

    # ------------
    # Train loop
    # ------------
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
            real_imgs = imgs.to(device)
            b_size = real_imgs.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #######################################

            # ------------
            # Train Discriminator
            # ------------
            discriminator.zero_grad()
            real_output = discriminator(real_imgs).view(-1)

            real_loss = adversarial_loss(real_output, label)
            real_loss.backward()
            D_x = real_output.mean().item()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            fake_imgs = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake_imgs.detach()).view(-1)

            fake_loss = adversarial_loss(output, label)
            fake_loss.backward()
            D_G_z1 = output.mean().item()

            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            #optimizer_G.zero_grad()
            generator.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_imgs).view(-1)

            g_loss = adversarial_loss(fake_output, label)
            #g_loss = adversarial_loss(fake_output, valid)

            g_loss.backward()
            D_G_z2 = fake_output.mean().item()
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

# ------------
# plot figure
# ------------
real_batch = next(iter(dataloader))[0]
with torch.no_grad():
    fake = generator(fixed_noise).detach().cpu()
fname = "results_conditional_gan/dataset_im_fin.png"
plot_results(fname, real_batch, fake)

# ------------
# plot learning curve
# ------------
d_loss_arr = np.array(d_loss_arr)
g_loss_arr = np.array(g_loss_arr)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss_arr,label="G")
plt.plot(d_loss_arr,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

plt.savefig ("results_conditional_gan/loss.png")
plt.show()
import itertools

import numpy as np
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchviz import make_dot

import torch.nn as nn
import time
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

channels = 1
img_size = 28
#img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
latent_dim = 100
num_classes = 10
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

    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.num_classes = num_classes

        # Hout = (Hin - 1)×stride − 2×padding + kernel_size

        self.main = nn.Sequential(
            # Hout = (Hin - 1)×stride − 2×padding + kernel_size
                                  # dim,    channels, kernel, stride, padding
            # Hin = 1, stride = 1, padding = 0,  kernel_size = 4
            # 4
            nn.ConvTranspose2d(latent_dim + num_classes, ngf * 8,  4,       1,       0, bias=False),
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

    def forward(self, input, labels):
        # embedding layer that transforms labels into dense vector representation
        # output shape is 2D tensor [batch_size, num_classes]
        label_embedding = self.label_emb(labels)

        # reshapes tensor to [batch_size, num_classes, 1,1]

        label_embedding = label_embedding.view(labels.size(0),
                                               self.num_classes,1,1)
        # concatenates along dimension 1, gen_input shape is
        # [batch_size, latent_dim + num_classes, 1,1]
        gen_input = torch.cat((input, label_embedding), 1)
        return self.main(gen_input)


class Discriminator(nn.Module):

    def __init__(self, num_classes, image_size):

        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.num_classes = num_classes
        self.image_size = image_size

        self.main = nn.Sequential(
            # input is ``(nc)x 28x28``
            nn.Conv2d(channels+num_classes, ndf, 4, 2, 1, bias=False),
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

    def forward(self, input, labels):

        label_embedding = self.label_emb(labels)

        label_embedding = label_embedding.view(labels.size(0),
                                               self.num_classes,1,1)

        label_embedding = label_embedding.expand(-1, -1,
                                                 self.image_size,
                                                 self.image_size)

        d_input = torch.cat((input, label_embedding), 1)

        return self.main(d_input)


class DatasetMNIST(Dataset):  # inherit abstract class - 'Dataset'

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image ad ndarray type (H, W, C)
        # be carefull for converting dtype to np.uint8 (Unsigned integer (0 to 255))

        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        if self.transform is not None:
            image = self.transform(image)

        return image, label



def Classifier(dataset, generator):
    # ------------
    # Generate test data set
    # ------------
    generator.eval()
    generated_samples = []
    generated_labels=[]
    num_samples_per_class = 1000

    for label in range(num_classes):

        noise = torch.randn(num_samples_per_class,
                        latent_dim, 1, 1,
                        device=device)

        #fixed_noise = torch.randn(64, latent_dim, device=device)  # Random noise
        #gen_labels = torch.randint(0, num_classes, (num_samples_per_class,), device=device)  # Random labels

        gen_labels = torch.full((num_samples_per_class,),
                            label, dtype = torch.long,
                            device=device)

        gen_images = generator(noise, gen_labels).detach().cpu().numpy()

        gen_images = gen_images.reshape(num_samples_per_class, -1)

       # print(gen_images.shape)
       # input()

        generated_samples.append(gen_images)
        generated_labels.extend([label]*num_samples_per_class)
    #generated_labels.append(gen_labels)

    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)

    print("shape")
    print(generated_samples.shape)
    print(generated_labels.shape)

    # ------------
    # Form training set from real images
    # ------------

    batch_size = 1  # suggested default, size of the batches
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, drop_last = True)

    real_images = []
    real_labels = []

    for i, (imgs, label_num) in enumerate(dataloader, 0):
        real_images.append(imgs[0])
        real_labels.append(label_num[0])

    real_images = np.array(real_images)
    real_labels = np.array(real_labels)

    train_indices = np.arange(0, 42000)
    #val_indices = np.arange(50000, 60000)

    real_train_images = real_images[0:42000]
    real_train_images = real_train_images.reshape(real_train_images.shape[0],-1)
    real_train_labels = real_labels[0:42000]

    #print(real_train_images.shape)
    #print(real_train_labels.shape)

    #real_val_images = real_images[val_indices]
    #real_val_labels = real_labels[val_indices]

    # ------------
    # Form training set from real images
    # ------------

    k = 5
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit (real_train_images, real_train_labels)
    predicted_labels = knn.predict(generated_samples)

    accuracy = np.mean(predicted_labels == generated_labels)

    print(f'Classification accuracy: {accuracy}')


if __name__ == '__main__':

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
    latent_dim = 100  # suggested default. dimensionality of the latent space

    num_classes = 10
    image_size = 28
    batch_size = 64  # suggested default, size of the batches

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if torch.cuda.is_available() and ngpu > 0:
        print("CUDA!")
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    adversarial_loss = torch.nn.BCELoss()

    generator = Generator(num_classes = num_classes).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(num_classes = num_classes,
                                  image_size = image_size).to(device)
    discriminator.apply(weights_init)

    train = pd.read_csv('data/train.csv')
    dataset = DatasetMNIST(file_path = 'data/train.csv',
                           transform = transforms.Compose( [transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]))

    Classifier(dataset, generator)

    # ------------
    # Alternatively we can download dataset from PyTorch
    # ------------

    #transform = transforms.Compose([
    #    transforms.Resize(image_size),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5,), (0.5,))
    #])

    #mnist_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    #dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    batch_size = 64  # suggested default, size of the batches
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, drop_last = True)

    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    real_label = 1
    fake_label = 0

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    # fixed_noise = torch.randn(64, latent_dim, device=device)  # Random noise
    fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    # Random labels

    #########################################
    # Show the  generator output before training
    nrows = 5
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))

    real_batch = next(iter(dataloader))
    real_batch_label = real_batch[1]
    real_batch_img = real_batch[0]

    #print("batch shape", real_batch.shape)
    #print ("data loader shape", dataloader.shape)

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    for nrow, ncol in itertools.product(range(ncols//2), range(nrows)):

        axes[ncol][nrow].imshow(real_batch_img.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow].axis('off')
        axes[ncol][nrow].set_title('real, ' + str(real_batch_label[ncol * nrow + nrow].item()))

        axes[ncol][nrow+2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow+2].axis('off')
        axes[ncol][nrow+2].set_title('fake, ' + str(fixed_labels[ncol * nrow + nrow].item()))

    plt.savefig ("results_conditional_gan/dataset_im_start.png")
    plt.show()

    #####################################
    #save the computation graph in folder \result
    #fake1 = generator(fixed_noise)
    #dot = make_dot(fake1, params=dict(generator.named_parameters()),
    #               show_attrs=False, show_saved=False)
    #dot.render("computation_graph_dicrim", format="png")
    #######################################

    n_epochs = 10
    iteration = 0

    d_loss_arr = []
    g_loss_arr = []

    for epoch in range(n_epochs):
        for i, (imgs, label_num) in enumerate(dataloader, 0):

            iteration += 1

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(real_label),
                             requires_grad=False)

            # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(imgs.size(0), 1).fill_(fake_label),
                            requires_grad=False)

            #######################################
            real_imgs = imgs.to(device)
            labels = label_num.to(device)

            #b_size = real_imgs.size(0)

            # ------------
            # Train Discriminator
            # ------------
            discriminator.zero_grad()

            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            valid_labels = labels
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            real_output = discriminator(real_imgs, valid_labels).view(-1)

            #print(real_output.shape, real_imgs.shape)

            real_loss = adversarial_loss(real_output, label)
            real_loss.backward()
            D_x = real_output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            #noise = torch.randn(batch_size, latent_dim, device=device)

            fake_imgs = generator(noise, fake_labels)
            label.fill_(fake_label)

            fake_output = discriminator(fake_imgs.detach(), fake_labels).view(-1)

            fake_loss = adversarial_loss(fake_output, label)
            fake_loss.backward()
            D_G_z1 = fake_output.mean().item()

            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            #optimizer_G.zero_grad()
            generator.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_imgs, fake_labels).view(-1)

            g_loss = adversarial_loss(fake_output, label)
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
                ncols = 4

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
                plt.suptitle('EPOCH : {} | BATCH(ITERATION) : {}'.format(epoch + 1, i + 1))

                with torch.no_grad():
                    fake = generator(fixed_noise, fixed_labels).detach().cpu()

                for nrow, ncol in itertools.product(range(ncols), range(nrows)):

                    axes[ncol][nrow].imshow(fake.permute(0, 2, 3, 1)[ncol*nrow + nrow], cmap='gray')
                    axes[ncol][nrow].axis('off')
                    axes[ncol][nrow].set_title('fake, ' + str(fixed_labels[ncol*nrow + nrow].item()))
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

    real_batch = next(iter(dataloader))
    real_batch_label = real_batch[1]
    real_batch_img = real_batch[0]
    # print("batch shape", real_batch.shape)
    # print ("data loader shape", dataloader.shape)

    generator.eval()
    with torch.no_grad():

        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    for nrow, ncol in itertools.product(range(ncols // 2), range(nrows)):

        axes[ncol][nrow].imshow(real_batch_img.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow].axis('off')
        axes[ncol][nrow].set_title('real, ' + str(real_batch_label[ncol * nrow + nrow].item()))


        axes[ncol][nrow + 2].imshow(fake.permute(0, 2, 3, 1)[ncol * nrow + nrow], cmap='gray')
        axes[ncol][nrow + 2].axis('off')
        axes[ncol][nrow + 2].set_title('fake, ' + str(fixed_labels[ncol * nrow + nrow].item()))

    plt.savefig ("results_conditional_gan/dataset_im_fin.png")

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

    plt.savefig ("results_conditional_gan/loss.png")
    plt.show()

    Classifier(dataset, generator)



import itertools

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
from torchviz import make_dot

import torch
import pandas as pd
import matplotlib.pyplot as plt

from models.conditional_gan import Generator, Discriminator, weights_init
from dataset_batch import DatasetMNIST
from classifier import Classifier
from plot_results import plot_results

def generate_train_test(dataset, generator, num_classes):
    # ------------
    # Generate test data set
    # ------------
    generator.eval()
    generated_samples = []
    generated_labels = []
    # num_samples_per_class = 1000

    for label in range(num_classes):
        noise = torch.randn(num_samples_per_class,
                            latent_dim, 1, 1,
                            device=device)

        gen_labels = torch.full((num_samples_per_class,),
                                label, dtype=torch.long,
                                device=device)

        gen_images = generator(noise, gen_labels).detach().cpu().numpy()

        gen_images = gen_images.reshape(num_samples_per_class, -1)

        generated_samples.append(gen_images)
        generated_labels.extend([label] * num_samples_per_class)

    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)

    # ------------
    # Form training set from real images
    # ------------

    batch_size = 1  # suggested default, size of the batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    real_images = []
    real_labels = []

    for i, (imgs, label_num) in enumerate(dataloader, 0):
        real_images.append(imgs[0])
        real_labels.append(label_num[0])

    real_images = np.array(real_images)
    real_labels = np.array(real_labels)

    train_indices = np.arange(0, 42000)

    real_train_images = real_images[0:42000]
    real_train_images = real_train_images.reshape(real_train_images.shape[0], -1)
    real_train_labels = real_labels[0:42000]

    return real_train_labels, real_train_images, generated_labels, generated_samples


if __name__ == '__main__':

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
    latent_dim = 100  # suggested default. dimensionality of the latent space

    num_classes = 10
    image_size = 28
    batch_size = 64  # suggested default, size of the batches

    batch_size = 64  # suggested default, size of the batches
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    real_label = 1
    fake_label = 0

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

    # --------
    # calculate accuracy of KNN classifier with untrained generator
    # --------
    num_samples_per_class = 1000

    real_train_labels, real_train_images, generated_labels, generated_samples = \
        generate_train_test(dataset, generator, num_classes)

    Classifier(real_train_labels, real_train_images, generated_labels, generated_samples)

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


    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # --------
    # plot untrained generator output
    # --------
    real_batch = next(iter(dataloader))[0]
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    fname = "results_conditional_gan/dataset_im_start.png"
    plot_results(fname, real_batch, fake)

    # --------
    # save the computation graph in folder
    # --------

    #fake1 = generator(fixed_noise)
    #dot = make_dot(fake1, params=dict(generator.named_parameters()),
    #               show_attrs=False, show_saved=False)
    #dot.render("computation_graph_dicrim", format="png")

    # --------
    # train loop
    # --------
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

    # --------
    # plot trained generator output
    # --------
    real_batch = next(iter(dataloader))[0]
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    fname = "results_conditional_gan/dataset_im_fin.png"
    plot_results(fname, real_batch, fake)

    # --------
    # plot learning curves
    # --------
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

    # --------
    # calculate accuracy of KNN classifier with trained generator
    # --------
    num_samples_per_class = 1000

    real_train_labels, real_train_images, generated_labels, generated_samples = \
        generate_train_test(dataset, generator, num_classes)

    Classifier(real_train_labels, real_train_images, generated_labels, generated_samples)



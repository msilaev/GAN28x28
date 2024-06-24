import torch.nn as nn
import torch

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

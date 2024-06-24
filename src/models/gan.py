import torch.nn as nn

channels = 1
img_size = 28
#img_shape = (channels, img_size, img_size)  # (Channels, Image Size(H), Image Size(W))
latent_dim = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Hout = (Hin - 1)×stride − 2×padding + kernel_size

        self.main = nn.Sequential(
            # Hout = (Hin - 1)×stride − 2×padding + kernel_size
                                  # dim,    channels, kernel, stride, padding
            # Hin = 1, stride = 1, padding = 0,  kernel_size = 4
            # 4
            nn.ConvTranspose2d(latent_dim, ngf * 8,  4,       1,       0, bias=False),
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

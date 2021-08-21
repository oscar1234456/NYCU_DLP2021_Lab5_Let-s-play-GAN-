import torch.nn as nn
import parameters
import torch.backends.cudnn as cudnn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(parameters.nz, parameters.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(parameters.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(parameters.ngf * 8, parameters.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( parameters.ngf * 4, parameters.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( parameters.ngf * 2, parameters.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( parameters.ngf, parameters.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(parameters.nc, parameters.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(parameters.ndf, parameters.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(parameters.ndf * 2, parameters.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(parameters.ndf * 4, parameters.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(parameters.ndf * 8, 1, 4, 1, 0, bias=False),
            #TODO: WGAN
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


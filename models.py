#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab5 Let's play GANs
#Date: 2021/08/21
#Subject: Implementing the cGAN model to generate pictures with label
#Email: oscarchen.cs10@nycu.edu.tw

##
import torch
import torch.nn as nn
import parameters
import torch.backends.cudnn as cudnn
#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ylabel = nn.Sequential(
            nn.Linear(24, 192),
            nn.ReLU(True)
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.yz = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(392, parameters.ngf * 8, 4, 1, 0, bias=False),
            # nn.ConvTranspose2d(124, parameters.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(parameters.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(parameters.ngf * 8, parameters.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( parameters.ngf * 4, parameters.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( parameters.ngf * 2, parameters.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( parameters.ngf, parameters.nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )


    def forward(self, z, y):
        z = self.yz(z.view(-1,100))
        y = self.ylabel(y)

        inp = torch.cat([z, y], dim=1)
        inp = inp.view(-1, 392, 1, 1)
        output = self.main(inp)
        return output  #size:N(128)*(c)3*64*64
        #myown:
        # condi_out = torch.cat([z, y.view(-1,24,1,1)], dim=1)
        # out = self.main(condi_out)
        # return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ylabel = nn.Sequential(
            nn.Linear(24, 64 * 64 * 1),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.Conv2d(parameters.nc+1, parameters.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),

            nn.Conv2d(parameters.ndf, parameters.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),

            nn.Conv2d(parameters.ndf * 2, parameters.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),

            nn.Conv2d(parameters.ndf * 4, parameters.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(parameters.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(),

            nn.Conv2d(parameters.ndf * 8, 1, 4, 1, 0, bias=False),
            #TODO: WGAN
            nn.Sigmoid()
        )


    def forward(self, input, condi):
        #normal version:
        y = self.ylabel(condi)
        y = y.view(-1, 1, 64, 64)
        inp = torch.cat([input, y], 1)
        output = self.main(inp)
        return output  #N(128)*1*1*1
        #myown:
        # output = self.main(input)
        # condi_input = torch.cat([output, condi.view(-1,24,1,1)], dim = 1)
        # new_output = self.conditionLayer(condi_input.view(-1,25))
        # return new_output
        #my own 2:
        # condi_out = self.ylabel(condi)
        # out = self.firstLayer(input)
        # out = torch.cat((out, condi_out.view(-1, parameters.ndf * 2, 16, 16)))
        # out = self.secondLayer(out)
        # return out




## Test Version
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         # self.ylabel = nn.Sequential(
#         #     nn.Linear(24, 192),
#         #     nn.ReLU(True)
#         #     # nn.LeakyReLU(0.2, inplace=True),
#         # )
#         #
#         # self.yz = nn.Sequential(
#         #     nn.Linear(100, 200),
#         #     nn.ReLU(True)
#         # )
#
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             # nn.ConvTranspose2d(392, parameters.ngf * 8, 4, 1, 0, bias=False),
#             nn.ConvTranspose2d(124, parameters.ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(parameters.ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(parameters.ngf * 8, parameters.ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( parameters.ngf * 4, parameters.ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( parameters.ngf * 2, parameters.ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( parameters.ngf, parameters.nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#
#     def forward(self, z, y):
#         # # mapping noise and label
#         # z = self.yz(z.view(-1,100))
#         # y = self.ylabel(y)
#         #
#         # # mapping concatenated input to the main generator network
#         # inp = torch.cat([z, y], dim=1)
#         # inp = inp.view(-1, 392, 1, 1)
#         # output = self.main(inp)
#         # return output  #size:N(128)*(c)3*64*64
#         #myown:
#         condi_out = torch.cat([z, y.view(-1,24,1,1)], dim=1)
#         out = self.main(condi_out)
#         return out
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.ylabel = nn.Sequential(
#             nn.Linear(24, parameters.ndf * 2* 16*16),
#             nn.ReLU(True)
#         )
#         # self.main = nn.Sequential(
#         #     # input is (nc) x 64 x 64
#         #     nn.Conv2d(parameters.nc+1, parameters.ndf, 4, 2, 1, bias=False),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf) x 32 x 32
#         #     nn.Conv2d(parameters.ndf, parameters.ndf * 2, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(parameters.ndf * 2),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf*2) x 16 x 16
#         #     nn.Conv2d(parameters.ndf * 2, parameters.ndf * 4, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(parameters.ndf * 4),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf*4) x 8 x 8
#         #     nn.Conv2d(parameters.ndf * 4, parameters.ndf * 8, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(parameters.ndf * 8),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf*8) x 4 x 4
#         #     nn.Conv2d(parameters.ndf * 8, 1, 4, 1, 0, bias=False),
#         #     #TODO: WGAN
#         #     nn.Sigmoid()
#         # )
#         # self.conditionLayer = nn.Sequential(
#         #     nn.Linear(1+24,1),
#         #     nn.Sigmoid()
#         # )
#
#         self.firstLayer = nn.Sequential(
#             nn.Conv2d(parameters.nc , parameters.ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(parameters.ndf, parameters.ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.secondLayer = nn.Sequential(
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(parameters.ndf * 2*2, parameters.ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(parameters.ndf * 4, parameters.ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(parameters.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(parameters.ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input, condi):
#         #li hung yi version:
#         # y = self.ylabel(condi)
#         # y = y.view(-1, 1, 64, 64)
#         # inp = torch.cat([input, y], 1)
#         # output = self.main(inp)
#         # # return output.view(-1, 1).squeeze(1)
#         # return output  #N(128)*1*1*1
#         #myown:
#         # output = self.main(input)
#         # condi_input = torch.cat([output, condi.view(-1,24,1,1)], dim = 1)
#         # new_output = self.conditionLayer(condi_input.view(-1,25))
#         # return new_output
#         #my own 2:
#         condi_out = self.ylabel(condi)
#         out = self.firstLayer(input) #N(128)*128*16*16
#         out = torch.cat((out, condi_out.view(-1, parameters.ndf * 2, 16, 16)), dim =1)
#         out = self.secondLayer(out)
#         return out



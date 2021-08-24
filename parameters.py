#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab5 Let's play GANs
#Date: 2021/08/21
#Subject: Implementing the cGAN model to generate pictures with label
#Email: oscarchen.cs10@nycu.edu.tw

workers = 4

batch_size = 128

image_size = 64

nc = 3

nz = 100

ngf = 64

ndf = 64

num_epochs = 50

lr = 0.0002

beta1 = 0.5
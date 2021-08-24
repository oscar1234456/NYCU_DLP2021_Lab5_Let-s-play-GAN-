#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab5 Let's play GANs
#Date: 2021/08/21
#Subject: Implementing the cGAN model to generate pictures with label
#Email: oscarchen.cs10@nycu.edu.tw

##
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab5 Let's play GANs
#Date: 2021/08/21
#Subject: Implementing the cGAN model to generate pictures with label
#Email: oscarchen.cs10@nycu.edu.tw

## import

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from dataset import ICLEVRLoader
import parameters
from  util import weights_init
from models import Generator, Discriminator
from trainner import train
import pickle

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

## DataLoader
train_data = ICLEVRLoader("./data", "./images")
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=parameters.batch_size,shuffle=True, num_workers=parameters.workers)
train_dataloader_dis = torch.utils.data.DataLoader(train_data, batch_size=parameters.batch_size,shuffle=True,num_workers=parameters.workers )

test_data = ICLEVRLoader("./data", "./images", mode="test")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=False, num_workers=parameters.workers)


##generatorNN
generatorNN = Generator().to(device)
generatorNN.apply(weights_init)
print(generatorNN)

##discriminatorNN
discriminatorNN = Discriminator().to(device)

discriminatorNN.apply(weights_init)

print(discriminatorNN)

##
criterion = nn.BCELoss()

# optimizer for DCGAN
# optimizerD = optim.Adam(discriminatorNN.parameters(), lr=0.0003, betas=(parameters.beta1, 0.999))
# optimizerG = optim.Adam(generatorNN.parameters(), lr=0.001, betas=(parameters.beta1, 0.999))
#optimizer for WGAN
optimizerD = optim.RMSprop(discriminatorNN.parameters(), lr=0.00005)
optimizerG = optim.RMSprop(generatorNN.parameters(), lr=0.0001)

##Training Process
#input: generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device
#DCGAN:
generator_final, discriminator_final,  img_list, (G_losses, D_losses) = train(generatorNN, discriminatorNN, parameters.num_epochs, parameters.nz, train_dataloader, criterion, optimizerD, optimizerG, device,test_dataloader)
#WGAN:
# generator_final, discriminator_final, img_list = train(generatorNN, discriminatorNN, parameters.num_epochs, parameters.nz, train_dataloader, criterion,optimizerD,optimizerG ,device, train_dataloader_dis, test_dataloader)
## Save model
torch.save(generator_final.state_dict(), './modelWeight/0822Test8/generator_weight1.pth')
torch.save(discriminator_final.state_dict(), './modelWeight/0822Test8/discriminator_weight1.pth')

##Save result
losses_data = (G_losses,D_losses)

with open('./modelWeight/0822Test8/pic.pickle', 'wb') as f:
    pickle.dump(img_list, f)

with open('./modelWeight/0822Test8/loss.pickle', 'wb') as f:
    pickle.dump(losses_data, f)
##
fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
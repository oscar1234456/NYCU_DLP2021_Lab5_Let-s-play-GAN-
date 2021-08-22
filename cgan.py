## import

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from dataset import ICLEVRLoader
import parameters
from  util import weights_init
from models import Generator, Discriminator
import time
from trainner import train
import pickle

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")

## DataLoader
train_data = ICLEVRLoader("./data", "./images")
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=parameters.batch_size,
                                         shuffle=True, num_workers=parameters.workers)
train_dataloader_dis = torch.utils.data.DataLoader(train_data, batch_size=parameters.batch_size,
                                         shuffle=True,num_workers=parameters.workers )

test_data = ICLEVRLoader("./data", "./images", mode="test")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                         shuffle=True, num_workers=parameters.workers)
##generatorNN
generatorNN = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
generatorNN.apply(weights_init)

print(generatorNN)

##discriminatorNN
discriminatorNN = Discriminator().to(device)

discriminatorNN.apply(weights_init)

print(discriminatorNN)

##
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminatorNN.parameters(), lr=parameters.lr, betas=(parameters.beta1, 0.999))
optimizerG = optim.Adam(generatorNN.parameters(), lr=parameters.lr, betas=(parameters.beta1, 0.999))
# optimizerD = optim.RMSprop(discriminatorNN.parameters(), lr=parameters.lr)
# optimizerG = optim.RMSprop(generatorNN.parameters(), lr=parameters.lr)

##Training Process
# generator_final, discriminator_final,  img_list= train(generatorNN, discriminatorNN, parameters.num_epochs, parameters.nz, train_dataloader, criterion, optimizerD, optimizerG, device,train_dataloader_dis)
#generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device
generator_final, discriminator_final,  img_list= train(generatorNN, discriminatorNN, parameters.num_epochs, parameters.nz, train_dataloader, criterion, optimizerD, optimizerG, device,test_dataloader)

## Save model
torch.save(generator_final.state_dict(), './modelWeight/0822Test4/generator_weight1.pth')
torch.save(discriminator_final.state_dict(), './modelWeight/0822Test4/discriminator_weight1.pth')


##Save result
with open('./modelWeight/0822Test4/pic.pickle', 'wb') as f:
    pickle.dump(img_list, f)
##
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
# HTML(ani.to_jshtml())

plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
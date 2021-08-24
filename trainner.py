#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab5 Let's play GANs
#Date: 2021/08/21
#Subject: Implementing the cGAN model to generate pictures with label
#Email: oscarchen.cs10@nycu.edu.tw

##
import random
import time
import torch
import parameters
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from dataset import labelLoader
from evaluator import evaluation_model
import copy

# manualSeed = 999
# #manualSeed = random.randint(1, 10000)
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

##DCGAN (Modified code from PyTorch Tutorial)
def train(generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device,test_dataloader):
    bestAcc=-999
    bestGWeight = copy.deepcopy(generator.state_dict())
    bestDWeight= copy.deepcopy(discriminator.state_dict())
    evalModel = evaluation_model()
    real_label = 1.
    fake_label = 0.
    since = time.time()

    img_list = []
    G_losses = []
    D_losses = []

    label_data = labelLoader("./data")

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(trainDataloader, 0):
            # discriminator.train()

            #Discriminator Trainning
            discriminator.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            breed  = data[1].to(device)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu, breed).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()


            noise = torch.randn(b_size, parameters.nz, 1, 1, device=device)
            fake = generator(noise, breed)
            label.fill_(fake_label)
            output = discriminator(fake.detach(), breed).view(-1)
            errD_fake = criterion(output, label)

            #real data/error condition
            breed_old = label_data.getRandomLabel(b_size)
            breed_old = breed_old.to(device)
            real_errCond_output = discriminator(real_cpu, breed_old).view(-1)
            real_errCond_err = criterion(real_errCond_output, label)

            totalError = real_errCond_err + errD_fake
            totalError.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            #Generator training
            generator.zero_grad()
            label.fill_(real_label)
            # discriminator.eval()
            output = discriminator(fake, breed).view(-1)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(trainDataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        print("Epoch Evaluating....")
        generator.eval()
        with torch.no_grad():
            for _, data in enumerate(test_dataloader, 0):
                noise_test = torch.randn(32, parameters.nz, 1, 1, device=device)
                fake = generator(noise_test, data[1].to(device)).detach().cpu()
                acc = evalModel.eval(fake.cuda(), data[1])
        generator.train()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()
        print(f"acc->{acc}")
        if acc > bestAcc:
            print("Best Model! Saved!")
            bestGWeight = copy.deepcopy(generator.state_dict())
            bestDWeight = copy.deepcopy(discriminator.state_dict())
            bestAcc = acc
            torch.save(bestGWeight, './modelWeight/0822Test7/generator_weight1.pth')
            torch.save(bestDWeight, './modelWeight/0822Test7/discriminator_weight1.pth')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    generator.load_state_dict(bestGWeight)
    discriminator.load_state_dict(bestDWeight)
    print(f"Best ACC:{bestAcc}")
    return generator, discriminator,  img_list,(G_losses,D_losses)

## WGAN
# def resetGrad(generator, discriminator):
#     generator.zero_grad()
#     discriminator.zero_grad()
#
#
# # for training discriminator
# def trainDiscriminator(generator, discriminator, trainDataloaderDis,optimizerD,device,label_data):
#     real_label = 1.
#     fake_label = 0.
#     print("trainDiscriminator....")
#     for epoch in range(3):
#         # random_ids = np.random.randint(len(trainDataloaderDis), size=64)
#         # batches = trainDataloaderDis[random_ids]
#         batches = next(iter(trainDataloaderDis))
#         fixed_noise = torch.randn(parameters.batch_size, parameters.nz, 1, 1, device=device)
#         real_cpu = batches[0].to(device)
#         b_size = real_cpu.size(0)
#         breed_old = label_data.getRandomLabel(b_size)
#         breed_old = breed_old.to(device)
#         G_sample = generator(fixed_noise, batches[1].to(device))
#         D_real = discriminator(real_cpu,batches[1].to(device)).view(-1)
#         D_fake = discriminator(G_sample,batches[1].to(device)).view(-1)
#         d_wrong_label = discriminator(real_cpu,breed_old).view(-1)
#         D_loss = -(torch.mean(D_real)-torch.mean(D_fake)+torch.mean(torch.log(1-d_wrong_label)))
#         D_loss.backward()
#         optimizerD.step()
#         for parm in discriminator.parameters():
#             parm.data.clamp_(-0.01, 0.01)
#         resetGrad(generator, discriminator)
#     return generator, discriminator, optimizerD, D_loss
#
# def train(generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device, trainDataloaderDis, test_dataloader):
#     bestAcc = -999
#     bestGWeight = copy.deepcopy(generator.state_dict())
#     bestDWeight = copy.deepcopy(discriminator.state_dict())
#     evalModel = evaluation_model()
#     label_data = labelLoader("./data")
#     real_label = 1.
#     fake_label = 0.
#     fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)
#     since = time.time()
#     # Training Loop
#
#     # Lists to keep track of progress
#     img_list = []
#     G_losses = []
#     D_losses = []
#     iters = 0
#
#
#     print("Starting Training Loop...")
#     # For each epoch
#     for epoch in range(num_epochs):
#         # For each batch in the dataloader
#         print("-"*50)
#         print(f"epoch:{epoch+1} is running...")
#
#         for i, data in enumerate(trainDataloader, 0):
#             generator, discriminator, optimizerD, D_loss = trainDiscriminator(generator, discriminator, trainDataloaderDis, optimizerD,device,label_data)
#             # input: generator, discriminator, trainDataloaderDis, optimizerD, device
#             # output: generator, discriminator, optimizerD, D_loss
#             b_size = data[0].size(0)
#             noise = torch.randn(b_size, parameters.nz, 1, 1, device=device)
#             G_sample = generator(noise, data[1].to(device))
#             D_fake = discriminator(G_sample, data[1].to(device)).view(-1)
#             G_loss = -torch.mean(D_fake)
#             G_loss.backward()
#             optimizerG.step()
#             resetGrad(generator, discriminator)
#
#             if i % 10 == 0:
#                 print('[%d/%d][%d/%d]'
#                         % (epoch, num_epochs, i, len(trainDataloader)))
#
#         print("Epoch Evaluating....")
#         generator.eval()
#         with torch.no_grad():
#             for _, data in enumerate(test_dataloader, 0):
#                 noise_test = torch.randn(32, parameters.nz, 1, 1, device=device)
#                 fake = generator(noise_test, data[1].to(device)).detach().cpu()
#                 acc = evalModel.eval(fake.cuda(), data[1])
#         generator.train()
#         img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
#         fig = plt.figure(figsize=(8, 8))
#         plt.axis("off")
#         plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
#         plt.show()
#         print(f"acc->{acc}")
#         if acc > bestAcc:
#             print("Best Model! Saved!")
#             bestGWeight = copy.deepcopy(generator.state_dict())
#             bestDWeight = copy.deepcopy(discriminator.state_dict())
#             bestAcc = acc
#             torch.save(bestGWeight, './modelWeight/0822Test8/generator_weight1.pth')
#             torch.save(bestDWeight, './modelWeight/0822Test8/discriminator_weight1.pth')
#
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#     generator.load_state_dict(bestGWeight)
#     discriminator.load_state_dict(bestDWeight)
#     print(f"Best ACC:{bestAcc}")
#     return generator, discriminator,  img_list
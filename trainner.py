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
# Set random seed for reproducibility
# manualSeed = 999
# #manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

def train(generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device,test_dataloader):
    bestAcc=-999
    bestGWeight = copy.deepcopy(generator.state_dict())
    bestDWeight= copy.deepcopy(discriminator.state_dict())
    evalModel = evaluation_model()
    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)
    since = time.time()
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    label_data = labelLoader("./data")
    # pickRandom = random.randint(1, 10)
    # for i in range(1, pickRandom):
    #     breed_old = next(iter(trainDataloader))[1]
    # breed_old = breed_old.to(device)
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(trainDataloader, 0):
            # discriminator.train()
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            breed  = data[1].to(device)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu, breed).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, parameters.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise, breed)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach(), breed).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            #real data/error condition
            breed_old = label_data.getRandomLabel(b_size)
            breed_old = breed_old.to(device)
            real_errCond_output = discriminator(real_cpu, breed_old).view(-1)
            real_errCond_err = criterion(real_errCond_output, label)
            #
            totalError = real_errCond_err + errD_fake
            totalError.backward()
            #errD_fake.backward()

            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            # breed_old = data[1].to(device)

            # (2) Update G network: maximize log(D(G(z)))

            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # discriminator.eval()
            output = discriminator(fake, breed).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(trainDataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
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
            torch.save(bestGWeight, './modelWeight/0822Test6/generator_weight1.pth')
            torch.save(bestDWeight, './modelWeight/0822Test6/discriminator_weight1.pth')
        # iters += 1

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
# def trainDiscriminator(generator, discriminator, trainDataloaderDis,optimizerD,device):
#     real_label = 1.
#     fake_label = 0.
#     print("trainDiscriminator....")
#     for epoch in range(5):
#         # random_ids = np.random.randint(len(trainDataloaderDis), size=64)
#         # batches = trainDataloaderDis[random_ids]
#         batches = next(iter(trainDataloaderDis))
#         fixed_noise = torch.randn(64, parameters.nz, 1, 1, device=device)
#         real_cpu = batches[0].to(device)
#         b_size = real_cpu.size(0)
#         G_sample = generator(fixed_noise)
#         D_real = discriminator(real_cpu).view(-1)
#         D_fake = discriminator(G_sample).view(-1)
#         D_loss = -(torch.mean(D_real)-torch.mean(D_fake))
#         D_loss.backward()
#         optimizerD.step()
#         for parm in discriminator.parameters():
#             parm.data.clamp_(-0.01, 0.01)
#         resetGrad(generator, discriminator)
#     return generator, discriminator, optimizerD, D_loss
#
# def train(generator, discriminator, num_epochs, latent_size, trainDataloader, criterion,optimizerD,optimizerG ,device, trainDataloaderDis):
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
#             generator, discriminator, optimizerD, D_loss = trainDiscriminator(generator, discriminator, trainDataloaderDis, optimizerD,device)
#             # input: generator, discriminator, trainDataloaderDis, optimizerD, device
#             # output: generator, discriminator, optimizerD, D_loss
#             noise = torch.randn(64, parameters.nz, 1, 1, device=device)
#             G_sample = generator(noise)
#             D_fake = discriminator(G_sample).view(-1)
#             G_loss = -torch.mean(D_fake)
#             G_loss.backward()
#             optimizerG.step()
#             resetGrad(generator, discriminator)
#
#             # Check how the generator is doing by saving G's output on fixed_noise
#             if (iters % 10 == 0) or ((epoch == num_epochs - 1) and (i == len(trainDataloader) - 1)):
#                 print('Iter-{}; Batch:{} ;D_loss: {}; G_loss: {}'.format(epoch + 1,i ,D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
#                 generator.eval()
#                 with torch.no_grad():
#                     fake = generator(fixed_noise).detach().cpu()
#                 img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
#                 fig = plt.figure(figsize=(8, 8))
#                 plt.axis("off")
#                 plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
#                 plt.show()
#                 generator.train()
#
#             iters += 1
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#     return generator, discriminator,  img_list
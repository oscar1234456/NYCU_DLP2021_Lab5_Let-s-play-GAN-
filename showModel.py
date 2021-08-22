##
import torch

from evaluator import evaluation_model
from models import Generator,Discriminator
import parameters
from dataset import ICLEVRLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##Parameters
#use parameters.py



##define model:
generator = Generator()
generator.to(device)


##
generator.load_state_dict(torch.load('modelWeight/0822Test4/generator_weight1.pth'))

##Dataset
test_data = ICLEVRLoader("./data", "./images", mode="test")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True, num_workers=parameters.workers)

#For Demo Day:
# test_data_new = ICLEVRLoader("./data", "./images", mode="test")
# test_dataloader_new = torch.utils.data.DataLoader(test_data_new, batch_size=*****,shuffle=True, num_workers=parameters.workers)

##
evalModel = evaluation_model()

## Original
print("Evaluating original TestData Start! Please  wait!")

generator.eval()
with torch.no_grad():
    for _, data in enumerate(test_dataloader, 0):
        noise_test = torch.randn(32, parameters.nz, 1, 1, device=device)
        fake = generator(noise_test, data[1].to(device)).detach().cpu()
        acc_ori = evalModel.eval(fake.cuda(), data[1])
generator.train()
print(f"original TestData ACC: {acc_ori}")

## New
print("Evaluating New TestData Start! Please  wait!")

# generator.eval()
# with torch.no_grad():
#     for _, data in enumerate(test_dataloader_new, 0):
#         noise_test = torch.randn(*****batch_size, parameters.nz, 1, 1, device=device)
#         fake = generator(noise_test, data[1].to(device)).detach().cpu()
#         acc_new = evalModel.eval(fake.cuda(), data[1])
# generator.train()
# print(f"new TestData ACC: {acc_new}")

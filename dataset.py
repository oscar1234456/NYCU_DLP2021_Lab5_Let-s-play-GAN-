import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import parameters
import torchvision.utils as vutils

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, image_folder, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        self.cond = cond
        self.num_classes = 24
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        data_transform = {
            "train": transforms.Compose(
                [
                    # Try Different Transform
                    transforms.Resize((parameters.image_size, parameters.image_size)),
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            ),
        }
        if self.mode == "train":
            img_path = self.image_folder + '/' + self.img_list[index]
            image = Image.open(img_path).convert('RGB')
            label = self.label_list[index]
            imageConvert = data_transform[self.mode](image)
        else:
            label = self.label_list[index]
            imageConvert = 0

        return imageConvert, torch.from_numpy(label).type(torch.float)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # label = get_iCLEVR_data("./data", "test")
    # # print(img)
    # print("--")
    # print(label)
    train_data = ICLEVRLoader("./data", "./images")
    # img, label = train_data[4]
    # plt.figure()
    # img_tran = img.numpy().transpose((1, 2, 0))
    # plt.imshow(img_tran)
    # plt.show()
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=parameters.batch_size,
                                             shuffle=True, num_workers=parameters.workers)
    print(len(dataloader))
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
    np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
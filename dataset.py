import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

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
    def __init__(self, root_folder, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
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
                    transforms.RandomRotation(degrees=(0, 360)),
                    # transforms.RandomResizedCrop(224),
                    # transforms.Resize(260),
                    # transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    # transforms.Normalize([0.4693, 0.3225, 0.2287], [0.1974, 0.1399, 0.1014])
                    transforms.Resize(224),
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]
            ),
        }
        img_path = self.root + '/' + self.img_name[index] + '.jpeg'
        image = Image.open(img_path).convert('RGB')
        label = self.label[index]
        imageConvert = data_transform[self.mode](image)
        return imageConvert, label


if __name__ == "__main__":
    label = get_iCLEVR_data("./data", "test")
    # print(img)
    print("--")
    print(label)
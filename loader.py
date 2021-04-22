from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import transforms
import torch
import cv2
import numpy as np


class MyDataset(Dataset):
    def __init__(self, mean,std,image_path,label_path,label_list,image_list,H = 512,W = 512):

        self.label_list = label_list
        self.image_list = image_list
        self.label_path = label_path
        self.image_path = image_path
        self.h = H
        self.w = W
        self.mean = mean
        self.std = std


        self.train_transforms_image = transforms.Compose([transforms.ToPILImage(),
        								   transforms.Resize((self.w, self.h)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(self.mean,self.std)])

        self.train_transforms_label = transforms.Compose([transforms.ToPILImage('F'),
        								   transforms.Resize((self.w, self.h),0)])


        self.test_transforms = transforms.Compose([transforms.Resize((self.w, self.h)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(self.mean,self.std)])

    
    def get_target_from_path(self, path):
        return torch.tensor([0])

    
    def __getitem__(self, index):
        
        x = torch.from_numpy(cv2.imread(self.image_path + '/' + self.image_list[index])).permute(2,0,1)
        y = torch.from_numpy(cv2.imread(self.label_path + '/' + self.label_list[index],cv2.IMREAD_GRAYSCALE)).float()

        x = self.train_transforms_image(x)
        y = self.train_transforms_label(y)

        y = torch.from_numpy(np.asarray(y))

        # x, y = self.transform(x,y)

        return x, y.unsqueeze(0)
    
    def __len__(self):
        return len(self.image_list)

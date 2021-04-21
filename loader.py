from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import os
from scipy import io
import paths.py as *
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self, mean,std,image_path,label_path,label_list,image_list,H = 256,W = 256):

        self.label_list = label_list
        self.image_list = image_list
        self.label_path = label_path
        self.image_path = image_path
        self.h = H
        self.w = W
        self.mean = mean
        self.std = std
    
    def get_target_from_path(self, path):
        # Implement your target extraction here
        return torch.tensor([0])

    
    def __getitem__(self, index):
        
        x = torch.from_numpy(cv2.imread(self.image_path + self.image_list[index])).permute(1,2,0) / 255
        y = torch.from_numpy(cv2.imread(self.label_path + self.label_list[index],cv2.cv2.IMREAD_GRAYSCALE))

        x = TF.normalize(x,self.mean,self.std)

        # x, y = self.transform(x,y)

        return x, y.unsqueeze(0)
    
    def __len__(self):
        return len(self.image_list)
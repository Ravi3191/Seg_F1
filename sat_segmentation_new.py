import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.transforms import transforms
import random
from PIL import Image
import os
import numpy as np
import torch.optim as optim
import torchvision
import cv2
import torchvision.transforms.functional as TF
from torchvision import models
import copy
from loader import MyDataset
from model import Net, convrelu
from iou_eval import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

NUM_CLASSES = 14
OPT_LEARNING_RATE_INIT 	= 1e-3

OPT_BETAS 		= (0.9, 0.999)
OPT_EPS_LOW 		= 1e-08
OPT_WEIGHT_DECAY 	= 1e-4

# image_path = '/home/tom/SUNRGBD_13class/SUNRGBD-train_images'
# label_path = '/home/tom/SUNRGBD_13class/train13labels'
# savedir = '/home/tom'

image_path = '/home/ravi/SUNRGBD/13Labels/SUNRGBD-train_images' 
label_path = '/home/ravi/SUNRGBD/13Labels/train13labels' 
savedir = '/home/ravi/SUNRGBD/13Labels'


image_list = os.listdir(image_path)
image_list.sort()

label_list = os.listdir(label_path)
label_list.sort()

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# class_weights = torch.tensor()
ARGS_NUM_EPOCHS = 50
tran_batch_size = val_batch_size = 32

Data = MyDataset(mean,std,image_path,label_path,label_list,image_list)
train_set, val_set = torch.utils.data.random_split(Data, [len(image_list) - 500, 500])
train_loader = DataLoader(train_set, batch_size=tran_batch_size,shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=val_batch_size,shuffle=True, num_workers=4)

class CrossEntropyLoss2d(torch.nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = torch.nn.NLLLoss(weight=weight)

	def forward(self, outputs, targets,mask=None):
		return self.loss(torch.nn.functional.log_softmax(outputs, dim=1),targets)

# weight = class_weights.cuda()
# criterion = CrossEntropyLoss2d(weight=weight)
criterion = CrossEntropyLoss2d()
iou_best = 0
model = Net(NUM_CLASSES,0.2).to(device)

optimizer = optim.Adam(
  model.parameters(),
  OPT_LEARNING_RATE_INIT,
  OPT_BETAS,
  eps = OPT_EPS_LOW,
  weight_decay = OPT_WEIGHT_DECAY 
  )

best_iou = 0

for epoch in range(ARGS_NUM_EPOCHS+1):
  print("\n ---------------- Epoch #", epoch, "------------------\n")
  epoch_loss = []
  iters = 0
  model.train()
  iouEvalTrain = iouEval(NUM_CLASSES,device)
  
  for step, (image,label) in enumerate(train_loader):

    iters += 1

    image = image.to(device)
    label = label.to(device)

    output = model(image)

    iouEvalTrain.addBatch(
      output.max(1)[1].unsqueeze(1).data,
      label.long().data
    )

    optimizer.zero_grad()
    loss = criterion(output,label[:,0].long())

    loss.backward()
    optimizer.step()
    epoch_loss.append(loss.item())
    
  
  avg_loss = sum(epoch_loss) / len(epoch_loss)
  iouTrain, iou_classes = iouEvalTrain.getIoU()

  print('[TRAINING] [Average loss]:{loss} [avg_iou]:{iou} [bg]:{bg} [roads]:{roads} [buildings]:{buildings} [vegetation]:{vegetation}'.format(
        loss = avg_loss,
        iou =  iouTrain))

  epoch_test_loss = []
  model.eval()
  iouEvalTest = iouEval(NUM_CLASSES,device)
  
  for step, (image,label) in enumerate(test_loader):

    image = image.to(device)
    label = label.to(device)

    output = model(image)

    iouEvalTest.addBatch(
      output.max(1)[1].unsqueeze(1).data,
      label.long().data
    )

    loss = criterion(output,label[:,0].long())
    epoch_test_loss.append(loss.item())
    
  
  avg_loss = sum(epoch_test_loss) / len(epoch_test_loss)
  iouTest, iou_classes = iouEvalTest.getIoU()

  

  print('[VALIDATION] [avg_val_loss]:{loss} [avg_iou]:{iou} [bg]:{bg} [roads]:{roads} [buildings]:{buildings} [vegetation]:{vegetation}'.format(
        loss = avg_loss,
        iou =  iouTest,
        bg = iou_classes[0]))
  
  if(iouTest > iou_best):
      iou_best = iouTest
      torch.save(model.state_dict(), savedir + '/model_best.pth')
      print('[SAVED] Best Model epoch:', epoch ,savedir +'/model_best.pth')
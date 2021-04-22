import torch
import cv2
import torch.nn as nn
from torchvision.transforms import transforms
from loader import MyDataset
from model import Net, convrelu
import os
import numpy as np

NUM_CLASSES = 14
model = Net(NUM_CLASSES,0.2)

def load_my_state_dict(model, state_dict):
    
    own_state = model.state_dict()
    
    for name, param in state_dict.items():
        if name not in own_state:
            print("[weight not copied for %s]"%(name)) 
            continue
        own_state[name].copy_(param)
    return model

model = load_my_state_dict(model,torch.load('/home/ravi/Ravi_D/Acadamic/UPENN/Fourth_sem/F1_10/model_best_200_balanced.pth'))

print('loaded model')

test_folder = '/home/ravi/Ravi_D/Acadamic/UPENN/Fourth_sem/F1_10/test_images/' 
test_images = os.listdir(test_folder)
print(len(test_images))
output_dir = '/home/ravi/Ravi_D/Acadamic/UPENN/Fourth_sem/F1_10/save_test/' 

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

w = 512
h = 512

test_transforms = transforms.Compose([transforms.ToPILImage(),
								  transforms.Resize((w, h)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean,std)])


labels = np.array([(0,0,0),(255,0,0),(0,255,0),(255,0,255),(75,0,175),(0,200,255)] )
video = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 3, (512,512))


for images in test_images:

	image = cv2.imread(test_folder + images)
	image = torch.from_numpy(image).permute(2,0,1)
	image = test_transforms(image).unsqueeze(0)
	print(image.shape)

	output = model(image)
	print('output shape',output.shape)

	label_out = output[0].max(0)[1].byte().data
	final_labels = torch.zeros((512,512))

	final_labels[label_out == 12] = 1 		#window
	final_labels[label_out == 13] = 1		#wall

	final_labels[label_out == 2] = 2 		#Books
	final_labels[label_out == 7] = 2		#Objects
	final_labels[label_out == 11] = 2 		#TV
	final_labels[label_out == 8] = 2		#Picture

	final_labels[label_out == 1] = 3 		#bed
	final_labels[label_out == 4] = 3		#chair
	final_labels[label_out == 6] = 3 		#Furniture
	final_labels[label_out == 9] = 3		#Sofa
	final_labels[label_out == 10] = 3 		#Table

	final_labels[label_out == 3] = 4 		#ceiling

	final_labels[label_out == 5] = 5 		#floor



	label_out = torch.Tensor(labels[final_labels.byte().data])
	save_dir = output_dir + images[:-3] + 'png'
	cv2.imwrite(save_dir,label_out.numpy())
	print('Saved to:', save_dir)


	video.write(label_out.numpy().astype(np.uint8))

video.release()
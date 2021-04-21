import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Net(nn.Module):
    def __init__(self, p = 0.35):
        super(Net,self).__init__()

        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample4 = nn.ConvTranspose2d(256,256,3,2,1,1)
        self.upsample3 = nn.ConvTranspose2d(128,128,3,2,1,1)
        self.upsample2 = nn.ConvTranspose2d(64,64,3,2,1,1)
        self.upsample1 = nn.ConvTranspose2d(64,64,3,2,1,1)
        self.upsample0 = nn.ConvTranspose2d(32,32,3,2,1,1)
        
        self.conv_up4 = convrelu(512 , 256 , 3 , 1)
        self.conv_up3 = convrelu(256 , 128, 3, 1)
        self.conv_up2 = convrelu(128, 64, 3, 1)
        self.conv_up1 = convrelu(64 , 64, 3, 1)
        self.conv_up0 = convrelu(64, 32, 3, 1)

        self.conv_original_size0 = convrelu(3, 32, 3, 1)
        self.conv_original_size1 = convrelu(32, 32, 3, 1)
        self.conv_original_size2 = convrelu(32, 16, 3, 1)

        self.conv_last = nn.Conv2d(16, 4, 1)
        self.dropout = nn.Dropout(p) 

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        #print(x_original.shape)
        layer0 = self.layer0(input) 
        #print(layer0.shape)
        layer1 = self.layer1(layer0) #dropout can be added here inside
        
        layer2 = self.layer2(layer1) #dropout can be added here inside
        
        layer3 = self.layer3(layer2) #dropout can be added here inside
          
        layer4 = self.layer4(layer3) #dropout can be added here inside
        
        layer4 = self.dropout(self.layer4_1x1(layer4)) 
        
        layer4 = self.conv_up4(layer4) #256
        
        x = self.upsample4(layer4) #256
        layer3 = self.layer3_1x1(layer3) # 256
        #print('first concat:',x.shape,layer3.shape)
        x = x + layer3 #torch.cat([x, layer3], dim=1) #256
        
        #print(x.shape)
        x = self.conv_up3(x) # 128
        #print(x.shape)
        x = self.upsample3(x) # 128
        layer2 = self.layer2_1x1(layer2) #128
        #print('second concat', x.shape,layer2.shape)
        x = x + layer2#torch.cat([x, layer2], dim=1) # 128
        x = self.conv_up2(x) #64

        x = self.upsample2(x) #64
        layer1 = self.layer1_1x1(layer1) #64

        x = x + layer1 #torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x) #64

        x = self.upsample1(x) #64
        layer0 = self.layer0_1x1(layer0) #64

        x =  x + layer0 #torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x) #32

        x = self.upsample0(x) #32

        x = x + x_original#torch.cat([x, x_original], dim=1) #32
        x = self.conv_original_size2(x) #16

        out = self.conv_last(x) #6

        return out
import math

import torch

from unet_parts import *
from torchsummary import summary
import torchvision.models as models
from torch.autograd import Variable

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,track_running=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64,track_running=track_running)
        self.down1 = Down(64, 128,track_running=track_running)
        self.down2 = Down(128, 256,track_running=track_running)
        self.down3 = Down(256, 512,track_running=track_running)
        self.down4 = Down(512, 512,track_running=track_running)
        self.up1 = Up(1024, 256, bilinear,track_running=track_running)
        self.up2 = Up(512, 128, bilinear,track_running=track_running)
        self.up3 = Up(256, 64, bilinear,track_running=track_running)
        self.up4 = Up(128, 64, bilinear,track_running=track_running)
        self.outc = OutConv(64, n_classes)

        self.factor = nn.Parameter(torch.ones(1)) #Variable(torch.ones(1,1),requires_grad=True)
        #self.factor=self.factor.to(device='cuda', dtype=torch.float32)
        self.factor.requires_grad = True

        self.factor_merge = nn.Parameter(torch.ones(1))
        self.factor_merge.requires_grad = True

    def forward_single(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #factor = 0.98 + self.factor*100
        logits = (1-self.factor)*logits + self.factor*x0/255.  #
        return logits

    def turn_angle(self, matr,angle):
        x0 = matr.shape[0]/2.
        y0 = matr.shape[0]/2.
        matr1 = torch.clone(matr)
        for i,j in [(i,j) for i in range(matr.shape[0]) for j in range(matr.shape[1])]:
            i1 = 72 + (i-72)*math.cos(angle) - (j-72)*math.sin(angle)
            j1 = 72 + (i-72)*math.sin(angle) + (j-72)*math.cos(angle)
            i1 = int(i1)
            j1 = int(j1)
            if i1 > 143 or j1 > 143: continue
            matr1[i1,j1] = matr[i,j]
        return matr

    def forward(self,x0):
        inp = x0[0,:,:,:]
        inp = torch.reshape(inp,(1,1,144,144))
        x = self.forward_single(inp)
        #for el in range(1,10):
        #    inp = x0[el, :, :, :]
        #    inp = torch.reshape(inp, (1, 1, 144, 144))
        #    x += self.factor_merge[el]*self.forward_single(inp)
        return x





if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    print(net)
    print("*** summary *** ")
    summary(net.cuda(), (1, 144, 144))
import numpy as np
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from numpy import loadtxt
import codecs
import math
from torch.autograd import Variable
import time
import gc
import SSIM_ke
import glob
import cv2
import os


class cus2(torch.nn.Module):

    def __init__(self):
        super(cus2,self).__init__()

    def forward(self, outputs, labels):
        lines = []
        chi = 0
        embedded = 0
        outputs=torch.reshape(outputs,shape=(144,144))
        labels= torch.reshape(labels, shape=(144, 144))
        chi += torch.sum(torch.abs(torch.subtract(labels,outputs)))
        #for i in range(outputs.shape[0]):
        #    for j in range(outputs.shape[1]):
        #        chi+=torch.abs((labels[i][j] - outputs[i][j]))

        try:
            for el in lines:
                el = str(el).replace('.','').rstrip(']').lstrip('[').rstrip().lstrip().replace('  ',' ')
                ind1 = int(el.split(' ')[0])
                ind2 = int(el.split(' ')[1])
                #print(ind1, ind2,labels[ind1][ind2])
                embedded+= torch.abs(labels[ind1][ind2] - outputs[ind1][ind2])
        except:
            embedded = 0
        #chi+= 100*embedded
        res = torch.sum(chi)
        return res

def validate_sample2(path0):
    label = []
    path = glob.glob(os.path.join(path0,'label/*.png'))
    for path in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        image = image / 255
        label.append(image)
    label = torch.tensor(label)
    label = label.to(device='cuda', dtype=torch.float32)
    image1 = []
    path = glob.glob(os.path.join(path0,'image/*.png'))
    for path in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        image1.append(image)
    image1 = torch.tensor(image1)
    image1 = image1.to(device='cuda', dtype=torch.float32)
    return (label,image1)

def validate_sample(path):
    label = []
    path = glob.glob(os.path.join(path,'*.png'))
    for path in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        image = image / 255
        label.append(image)
    label = torch.tensor(label)
    label = label.to(device='cuda', dtype=torch.float32)
    return label




def train_nn(Net, device, data_path, epochs=10, batch_size=4, lr=0.0001,weight_decay=0.0, momentum=0.9,alpha=0.9,shuffle=False, centered = False):

    isbi_dataset = Data_Loader(data_path)
    print('data_path = ',data_path)
    print('isbi_dataset = ',isbi_dataset)
    file = open("history_loss.dat","w")
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    criterion = nn.L1Loss() #nn.L1Loss()
    best_loss = float('inf')
    cal = 0
    optimizer = optim.Adam(Net.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    for epoch in range(epochs):
        for image, label, label_emb in train_loader:
            Net.train()
            learn_rate = scheduler.get_lr();
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            label_emb = label_emb.to(device=device, dtype=torch.float32)
            pred = Net(image)
            label = label[0]
            label = torch.reshape(label,(1,1,144,144))
            loss = criterion(pred, label)
            out = '';
            out += str(learn_rate) + f' {Net.UNet.factor.item()} ' + str(loss.item()) + ' '
            if loss.item() < best_loss:
                best_loss = loss
                torch.save(Net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()
            out += '\n'
            file.write(out)
            print(time.strftime("%a, %d %b %Y %H:%M", time.localtime()),' ',out, end='')
            cal+=1
            scheduler.step()
    file.close()

import numpy as np
from unet_model import UNet
from LSTM_UNET import LSTM_UNET
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from train import *
from gen_fig_2 import *
from shutil import copyfile
from predict import *
import os

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #Net = UNet(n_channels=1, n_classes=1, track_running=True)
    Net = LSTM_UNET()
    Net.to(device=device)
    Net.apply(weights_init_uniform_rule)

    #

    #train_nn(Net, device, 'data/', epochs=3, batch_size=10)
    draw_loss_from_file("history_loss.dat")
    predict("data/","best_model.pth")
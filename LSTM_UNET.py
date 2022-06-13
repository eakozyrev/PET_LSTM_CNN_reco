import torch.nn as nn
import torch.nn.functional as f
from unet_model import UNet
from LSTM import RNN

class LSTM_UNET(nn.Module):

    def __init__(self):
        super().__init__()
        self.UNet = UNet(n_channels=1, n_classes=1)
        self.rnn = RNN()
        self.linear = nn.Linear(64, 10)

    def forward(self,x):

        rout = self.rnn(x)
        #print(self.rnn)
        return  rout
        cout = self.UNet(rout)
        #print("Model parameters: ")
        #for par in self.rnn.parameters():
        #    print(par)
        #print("---------------------------")
        return cout
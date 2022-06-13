from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

class RNN(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
        )
       # self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(10,144,144)
        x = torch.transpose(x,0,2)
        rin = torch.reshape(x,(144*144,10,1))
        #rin/=255
        _, (a, r_out) = self.rnn(rin, None)

        #print(r_out,a,b)
        out = torch.reshape(r_out,(1,1,144,144))
        out = torch.transpose(out,2,3)
        # -----------
        #pred = np.array(out.data[0])[0]
        #plt.imshow(pred)
        #plt.show()
        #cv2.imwrite('1.png', pred)
        # ------------
        return out
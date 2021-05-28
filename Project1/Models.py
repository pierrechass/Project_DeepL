# MODELS : 

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F


####################################################################################################
"""
This model is our simpler model with two convolution and two linear layers (no dropout, no batch normalization, no auxiliary loss and no weight sharing)
This model is trained rapidely but give low performance. 

Performance with epoch=25, batch size=100, optimizer=Adadelta, criterion=CrossEntropyLoss, learning rate=1 ,number of run=10 : 

mean test error :   24,3 %
standard deviation : 3.97% 
computation time per training : 4.3 s

"""

class Netsimple(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x

"""
This model is our complex model with two convolution and three our 4 linear layers depending on which auxiliary loss method we want to use. 
It is a modulable model where we can select the kernel size of our convolution, if we want to add dropout and batchnorm. 
There is also wheight sharing (same convolution use for both images so that we extract the same features for both). 
There is two method for the auxiliary loss depending at which layer we want to implement it. 

Performance with epoch=25, batch size=100, optimizer=Adadelta, criterion=CrossEntropyLoss, learning rate=1 ,number of run=10, dropout, batchnorm,  auxiliaray loss (method 2) : 

mean test error :   2.27 %
standard deviation : 0.7134% 
computation time per training : 11.1 s
"""
class Netfinal(nn.Module):


    def __init__(self, nb_hidden, batchnorm=True, dropout=True, kernel_size=3, aux_loss_method=1):
        super().__init__()

        # design the convolution layers so that we have at the output of the second convolution a tensor of size [N,64,2,2]
        if (kernel_size)%2==1:
            self.conv1 = nn.Conv2d(1, 32,kernel_size = kernel_size,
                               padding = (kernel_size - 2) // 2)
            self.conv2 = nn.Conv2d(32, 64,kernel_size = kernel_size,
                        padding = (kernel_size - 2) // 2)


        else : 
            self.conv1 = nn.Conv2d(1, 32,kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)
            self.conv2 = nn.Conv2d(32, 64,kernel_size = kernel_size,
                        padding = (kernel_size - 1) // 2)
                        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(nb_hidden)
        self.bn4 = nn.BatchNorm1d(200)

        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
     
        self.dropout2 = nn.Dropout(p=0.2) #0.8   0.6
        self.dropout1 = nn.Dropout(p=0.1) #0.3  0.5


        self.softmax = nn.Softmax(dim=1)

        # part of the neural network to find the target
        self.fc3 = nn.Linear(512, 200)
        self.fc4 = nn.Linear(20,200)
        self.fc5 = nn.Linear(200, 2)
        
        # check if we want to add batchnorm, dropout and auxloss : 
        self.batchnorm=batchnorm
        self.dropout=dropout
        self.aux_loss_method=aux_loss_method

    def forward(self, x):
        # x is [N,2,14,14] and we want to get x1 = [N,1,14,14] and x2= [N,1,14,14] so that wwe classify one image at a time and not
        # the two images (Weightsharing)
        x1=x[:,0,:,:]
        x1=torch.unsqueeze(x1, 1) # to go from [N,14,14] to [N,1,14,14]
        x2=x[:,1,:,:]
        x2=torch.unsqueeze(x2, 1)

        # for x1  :
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2))
        if self.batchnorm : x1 = self.bn1(x1)
        if self.dropout : x1 = self.dropout1(x1)
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2))
        if self.batchnorm : x1 = self.bn2(x1)
        if self.dropout : x1 = self.dropout2(x1)
        x1class = F.relu(self.fc1(x1.view(-1, 256)))

        x1class = self.fc2(x1class)
        x1class = self.softmax(x1class)
       
        # for x2 : 
        x2 = F.relu(F.max_pool2d(self.conv1(x2), kernel_size=2))
        if self.batchnorm :  x2 = self.bn1(x2)
        if self.dropout : x2 = self.dropout1(x2)
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2))
        if self.batchnorm : x2 = self.bn2(x2)
        if self.dropout : x2 = self.dropout2(x2)
        x2class = F.relu(self.fc1(x2.view(-1, 256)))
  
        x2class = self.fc2(x2class)
        x2class = self.softmax(x2class)
        
        # layers to find the target : 
        if self.aux_loss_method==1: 
            y=torch.cat((x1,x2), 1)
            y=F.relu(self.fc3(y.view(-1, 512)))
            if self.batchnorm : y=self.bn4(y) 

        else :
            y=torch.cat((x1class,x2class), 1)
            y=F.relu(self.fc4(y.view(-1, 20)))

        y=self.fc5(y)
        #y=self.softmax(y)

        # the output will be x1 [N,10] and x2 [N,10] corresponding to the probability of the number on each images
        # and y [N,2] corresponding to the probability of which image is the biggest
        return x1class,x2class,y
######################################################################

"""
ResNet model using a ResNet block similar to the one use in exercice 6 in the course. We can select the number of Resnet block we want, 
the size of the kernel and the number of channels in our resnet block. 

Performance with epoch=25, batch size=100, optimizer=Adadelta, criterion=CrossEntropyLoss, learning rate=1 ,number of run=10 : 

mean test error :   18,15 %
standard deviation : 2.2634% 
computation time per training : 151 s
"""

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size,
                 skip_connections = True, batch_normalization = True):
        super().__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.relu(y)

        return y
    
    

class ResNet(nn.Module):

    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size = 3, nb_classes = 2,
                 skip_connections = True, batch_normalization = True):
        super().__init__()

        self.conv = nn.Conv2d(2, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size, skip_connections, batch_normalization)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(nb_channels, nb_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.resnet_blocks(x)
       
        x = F.avg_pool2d(x, 14).view(x.size(0), -1)
        x = self.fc(x)
        return x
######################################################################

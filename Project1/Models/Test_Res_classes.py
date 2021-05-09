# The Test_Res file give us bad result. So in this file, we try to see if we have better results using res block for classification

import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy

N=1000
[train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)


######################################################################

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
                 kernel_size = 3, nb_classes = 10,
                 skip_connections = True, batch_normalization = True):
        super().__init__()

        self.conv = nn.Conv2d(1, nb_channels,
                              kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(nb_channels)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size, skip_connections, batch_normalization)
              for _ in range(nb_residual_blocks))
        )

        self.fc = nn.Linear(nb_channels, nb_classes)

    def forward(self, x):
        # x is [N,2,14,14] and we want to get x1 = [N,1,14,14] and x2= [N,1,14,14] so that wwe classify one image at a time and not
        # the two images (Weightsharing)
        x1=x[:,0,:,:]
        x1=torch.unsqueeze(x1, 1)
        x2=x[:,1,:,:]
        x2=torch.unsqueeze(x2, 1)
       
       # for x1 : 
        x1 = F.relu(self.bn(self.conv(x1)))
        x1 = self.resnet_blocks(x1)
        x1 = F.avg_pool2d(x1, 14).view(x1.size(0), -1)
        x1 = self.fc(x1)
        
        # for x2 : 
        x2 = F.relu(self.bn(self.conv(x2)))
        x2 = self.resnet_blocks(x2)
        x2 = F.avg_pool2d(x2, 14).view(x2.size(0), -1)
        x2 = self.fc(x2)
        
        return x1,x2


def train_model(model, train_input, train_classes, mini_batch_size, nb_epochs = 50):
    #criterion = nn.MSELoss()
    model.train()
    eta = 1e-2
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(model.parameters(), lr = eta)
    
    
    
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output1,output2 = model(train_input.narrow(0, b, mini_batch_size))

            # extract the true classes of each image (the first and second images of the pair)
            train_classes1=train_classes[:,0]
            train_classes2=train_classes[:,1]

            loss1 = criterion(output1, train_classes1.narrow(0, b, mini_batch_size))
            loss2 = criterion(output2, train_classes2.narrow(0, b, mini_batch_size))
            loss=loss1+loss2

            acc_loss = acc_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        # print(e, acc_loss)

def compute_nb_errors(model, data_input, data_classes):
    model.eval()
    
    nb_data_errors_classes1 = 0
    nb_data_errors_classes2 = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        # extract the predicted classes and target
        x1,x2 = model(data_input.narrow(0, b, mini_batch_size))

        _, predicted_classes1 = torch.max(x1, 1)
        _, predicted_classes2 = torch.max(x2, 1)

        # extract the true classes
        data_classes1=data_classes[:,0]
        data_classes2=data_classes[:,1]

        for k in range(mini_batch_size):
            if data_classes1[b+k] != predicted_classes1[k]:
                nb_data_errors_classes1 = nb_data_errors_classes1 + 1
            if data_classes2[b+k] != predicted_classes2[k]:
                nb_data_errors_classes2 = nb_data_errors_classes2 + 1

    return nb_data_errors_classes1, nb_data_errors_classes2


######################################################################

mini_batch_size = 100
mean_error= numpy.zeros(10)

for k in range(10):
    model = ResNet(nb_residual_blocks = 15, nb_channels = 10,
                   kernel_size = 3, nb_classes = 10,
                   skip_connections = True, batch_normalization = True) # Large residual blocks and number of channel -> gives 0 errors for the train but 25% for the test set
    train_model(model, train_input, train_classes, mini_batch_size)      # small residual blocks give bad result for training and testing 
    nb_test_errors1, nb_test_errors2 = compute_nb_errors(model, test_input, test_classes)
    print('test error class1 {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors1) / test_input.size(0),
                                                      nb_test_errors1, test_input.size(0)))
    print('test error class2 {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors2) / test_input.size(0),
                                                      nb_test_errors2, test_input.size(0)))                                            
                                                    


    nb_train_errors1, nb_train_errors2 = compute_nb_errors(model, train_input, train_classes)
    print('train error class1 {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors1) / train_input.size(0),
                                                      nb_train_errors1, train_input.size(0)))
    print('train error class2 {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors2) / train_input.size(0),
                                                      nb_train_errors2, train_input.size(0)))



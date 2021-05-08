# FIND CLASSES : 


"""
The aim of this python file is to find the best neural network to find the class of the two images (the number on the images). The 
network will later be add in the Test6.py to classify the different images. 
"""

import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy

N=1000
[train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)

######################################################################

class Net(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
     
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.3)

    def forward(self, x):
        
        # x is [N,2,14,14] and we want to get x1 = [N,1,14,14] and x2= [N,1,14,14] so that wwe classify one image at a time and not
        # the two images (Weightsharing)
        x1=x[:,0,:,:]
        x1=torch.unsqueeze(x1, 1)
        x2=x[:,1,:,:]
        x2=torch.unsqueeze(x2, 1)

        # We implement the same neural network for x1 and x2 (weightsharing)  :
        # for x1  :
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2))
        x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2))
        x1 = self.bn2(x1)
        x1 = F.relu(self.fc1(x1.view(-1, 256)))
        x1 = self.dropout2(x1)
        x1 = self.fc2(x1)

        # for x2 : 
        x2 = F.relu(F.max_pool2d(self.conv1(x2), kernel_size=2))
        x2 = self.bn1(x2)
        x2 = self.dropout1(x2)
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2))
        x2 = self.bn2(x2)
        x2 = F.relu(self.fc1(x2.view(-1, 256)))
        x2 = self.dropout2(x2)   
        x2 = self.fc2(x2)

        # the output will be x1 [N,10] and x2 [N,10] corresponding to the probability of the number on each images
        return x1,x2

######################################################################



def train_model(model, train_input, train_classes, mini_batch_size, nb_epochs = 25):
    model.train() # put in train mode (take into account dropout and batchnormalization)

    criterion = nn.CrossEntropyLoss()
    eta = 1
    optimizer = torch.optim.Adadelta(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):

            class1,class2 = model(train_input.narrow(0, b, mini_batch_size))

            # extract the true classes of each image (the first and second images of the pair)
            train_classes1=train_classes[:,0] 
            train_classes2=train_classes[:,1]

            
            loss2 = criterion(class1, train_classes1.narrow(0, b, mini_batch_size)) # Loss for the first image in the pair
            loss3 = criterion(class2, train_classes2.narrow(0, b, mini_batch_size)) # loss on the second image in the pair 
            loss=loss3+loss2

            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
#######################################################################################################

#######################################################################################################


def compute_nb_errors(model, data_input,data_classes):
    
    model.eval() # put in eval mode (no dropout and batchnormalization)

    nb_data_errors_target = 0
    nb_data_errors_classes1 = 0
    nb_data_errors_classes2 = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        x1,x2 = model(data_input.narrow(0, b, mini_batch_size))
        
        # extract the predicted classes 
        _, predicted_classes1 = torch.max(x1, 1) 
        _, predicted_classes2 = torch.max(x2, 1)

        # extract the true classes
        data_classes1=data_classes[:,0]
        data_classes2=data_classes[:,1]

        # compute the numbre of error 
        for k in range(mini_batch_size):
            
            if data_classes1[b+k] != predicted_classes1[k]:
                nb_data_errors_classes1 = nb_data_errors_classes1 + 1
            if data_classes2[b+k] != predicted_classes2[k]:
                nb_data_errors_classes2 = nb_data_errors_classes2 + 1
    return nb_data_errors_classes1,nb_data_errors_classes2

######################################################################

# Main code : 

mini_batch_size = 100

mean_test_class1=numpy.zeros(10)
mean_test_class2=numpy.zeros(10)
mean_train_class1=numpy.zeros(10)
mean_train_class2=numpy.zeros(10) # tensor use to calculate the mean error on 10 run :

for k in range(10):  # run 10 time our algorithm to get a better approximation of the performance of our model 
    # initialize our model 
    model = Net(200) 
    
    # train our model
    train_model(model, train_input, train_classes, mini_batch_size)   
                                  
    # compute number of errors in our testing set
    error_class1, error_class2 = compute_nb_errors(model, test_input, test_classes)                              

    print('test error class1 {:0.2f}% {:d}/{:d}'.format((100 * error_class1) / test_input.size(0),
                                                      error_class1, test_input.size(0)))
    print('test error class2 {:0.2f}% {:d}/{:d}'.format((100 * error_class2) / test_input.size(0),
                                                      error_class2, test_input.size(0)))
    mean_test_class1[k]=error_class1
    mean_test_class2[k]=error_class2

    error_class1, error_class2 = compute_nb_errors(model, train_input, train_classes)                              

    print('train error class1 {:0.2f}% {:d}/{:d}'.format((100 * error_class1) / train_input.size(0),
                                                      error_class1, train_input.size(0)))
    print('train error class2 {:0.2f}% {:d}/{:d}'.format((100 * error_class2) / train_input.size(0),
                                                      error_class2, train_input.size(0)))
    mean_train_class1[k]=error_class1
    mean_train_class2[k]=error_class2

# compute the mean error on 10 run : 

print('mean test class1', numpy.mean(mean_test_class1)/test_input.size(0)*100)
print('mean test class2', numpy.mean(mean_test_class2)/test_target.size(0)*100)
print('mean train class1', numpy.mean(mean_train_class1)/train_input.size(0)*100)
print('mean train class2', numpy.mean(mean_train_class2)/train_target.size(0)*100)



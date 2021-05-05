# Projet 1

"""
For this first model we use only train_targets (and not train classes) to train our model. This model receive in input
a N*2*14*14 tensor corresponding to N sample of 2 images of size 14*14 and it will give as output which image has the 
biggest number

With the current parameter below we obtain the best result for the testing set but we have a lot of overfitting...
"""


import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy

# Get the train set and the test set (each with 1000 samples)
N=1000
[train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)


######################################################################################

# similar module as in serie 4 

    # BEST FOR THE MOMENT : (BUT A LOT OF OVERFITTING)
    # eta=1
    # optimizer: Adadelta 
    # nb_hidden : 200
    # score  :  19.3199 testing set    (with conv(2,32), conv(32,64))
    #           0.00    training set  
    # 
    #           20.51   testing set    (with conv(2,32), conv(32,64))
    #           1.63    training set

    # criterion : nn.crossEntropyLoss()

    # WITH LESS OVERFITTING 
    # eta=1e-2
    # optimizer: SGD 
    # nb_hidden : 20
    # score  :  25.41   testing set    (with conv(2,32), conv(32,64))
    #           11.23    training set  
    # 
    #           24.49   testing set    (with conv(2,16), conv(16,32)) -> faster computation and similar results
    #           11.87   training set

    # criterion : nn.crossEntropyLoss()


class Net(nn.Module):
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

######################################################################


def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 25):
    
    criterion = nn.CrossEntropyLoss()
    eta = 1
    optimizer = torch.optim.Adadelta (model.parameters(), lr = eta)
    for e in range(nb_epochs):
        acc_loss = 0

        # compute the training using mini_bach_size : 

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(e, acc_loss)   # to check evolution of the loss 

def compute_nb_errors(model, data_input, data_target):
    # function use to compute the number of error with this module 

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

######################################################################


# Main code : 

mini_batch_size = 100

mean_error_test=numpy.zeros(10) # tensor use to calculate the mean error on 10 run :
mean_error_train=numpy.zeros(10)
for k in range(10):  # run 10 time our algorithm to get a better approximation of the performance of our model 
    # initialize our model 
    model = Net(200)                                                                                
    # train our model
    train_model(model, train_input, train_target, mini_batch_size)                                  
    # compute number of errors in our testing set
    nb_test_errors = compute_nb_errors(model, test_input, test_target)                              
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    
    # compute number of errors in our testing set
    nb_train_errors = compute_nb_errors(model, train_input, train_target)                              
    print('train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))
    mean_error_test[k]=nb_test_errors
    mean_error_train[k]=nb_train_errors
print('mean test errors',numpy.mean(mean_error_test)/test_input.size(0)*100)
print('mean train errors',numpy.mean(mean_error_train)/train_input.size(0)*100)


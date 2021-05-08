# TEST ON AN EXAMPLE ON THE NET : 

# Projet 1

"""
For this model we test a model found on the internet. This model receive in input
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

train_input=train_input.view(-1,1,14,14)  #from [1000,2,14,14] to [2000,1,14,14]
train_classes=train_classes.view(-1)      #from [1000,2] to [2000]  
#print(train_input.size())

test_input=test_input.view(-1,1,14,14)    #from [1000,2,14,14] to [2000,1,14,14]
test_classes=test_classes.view(-1)        #from [1000,2] to [2000]  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

######################################################################

# function to train our model (similar to the one in serie 4)
def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 25):
    model.train()
    criterion = nn.CrossEntropyLoss()
    eta = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):

            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#######################################################################################################

# function to compute the number of wrong estimated number (error for the number estimated)
def compute_nb_errors_classes(model, data_input, data_target):
    model.eval()
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)

    
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#######################################################################################################

# function to compute the number of errors for the target (error for the choice of the biggest number)
def compute_nb_errors_target(model, data_input, data_target):
    model.eval()
    nb_data_errors = 0
    
    
    output = model(data_input)
    _, predicted_classes = torch.max(output, 1)
    A=data_target.size(0)
    
    for k in range(A):
        if predicted_classes[2*k]>predicted_classes[k*2+1]:
            sol=torch.tensor(0)
        else:
            sol=torch.tensor(1)
        #print(data_target[k])
        #print('sol',sol)
        if data_target[k] != sol:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#######################################################################################################

# MAIN Function : 

mini_batch_size = 100
mean_error_classes_test=numpy.zeros(10)  # tensor use to calculate the mean error on 10 run :
mean_error_target_test=numpy.zeros(10)  
mean_error_classes_train=numpy.zeros(10)  # tensor use to calculate the mean error on 10 run :
mean_error_target_train=numpy.zeros(10)  

for k in range(10):                 # run 10 time our algorithm to get a better approximation of the performance of our model
    # Initialize and train our model
    model = Net()  
    train_model(model, train_input, train_classes, mini_batch_size)

    # Check performance on the testing set :              
    nb_test_errors_classes = compute_nb_errors_classes(model, test_input, test_classes)
    print('test error classes Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors_classes) / test_input.size(0),
                                                      nb_test_errors_classes, test_input.size(0)))
    mean_error_classes_test[k]=nb_test_errors_classes
    
    nb_test_errors_target = compute_nb_errors_target(model, test_input, test_target)
    print('test error target Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors_target) / test_target.size(0),
                                                      nb_test_errors_target, test_target.size(0)))
    mean_error_target_test[k]=nb_test_errors_target


    # Check performance on the training set : 
    nb_train_errors_classes = compute_nb_errors_classes(model, train_input, train_classes)
    print('train error classes Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors_classes) / train_input.size(0),
                                                      nb_train_errors_classes, train_input.size(0)))
    mean_error_classes_train[k]=nb_train_errors_classes
    
    nb_train_errors_target = compute_nb_errors_target(model, train_input, train_target)
    print('train error target Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors_target) / train_target.size(0),
                                                      nb_train_errors_target, train_target.size(0)))
    mean_error_target_train[k]=nb_train_errors_target    

print('mean test error classes', numpy.mean(mean_error_classes_test)/test_input.size(0)*100)
print('mean test error target', numpy.mean(mean_error_target_test)/test_target.size(0)*100)
print('mean train error classes', numpy.mean(mean_error_classes_train)/train_input.size(0)*100)
print('mean train error target', numpy.mean(mean_error_target_train)/train_target.size(0)*100)

######################################################################################################
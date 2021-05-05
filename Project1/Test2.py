# Projet 1

import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy

N=1000
[train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)
print(train_input.size())
train_input=train_input.view(-1,1,14,14)  #2*train_input.size(0)
train_classes=train_classes.view(-1)
print(train_input.size())
test_input=test_input.view(-1,1,14,14)
test_classes=test_classes.view(-1)
#print(train_target)
#print(train_input[1,1,:,:])


######################################################################

class Net(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

######################################################################

def create_shallow_model():
    return nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 25):
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # BEST FOR THE MOMENT :
    # eta=1
    # optimizer: Adadelta 
    # score  : 19.3199

    eta = 1
    optimizer = torch.optim.Adadelta(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):

            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            #print('output',output)
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()
            """
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad
            """
        # print(e, acc_loss)
def train_model2(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 25

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()   # why not optimizer 0 grad?? 
            loss.backward()
            optimizer.step()

######################################################################
def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)

    
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#######################################################################################################

mini_batch_size = 100
mean_error=numpy.zeros(10)
mean_error2=numpy.zeros(10)
for k in range(10):
    model = Net(200)
    """
    train_model(model, train_input, train_classes, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_classes)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    mean_error[k]=nb_test_errors
    """

    model2 = create_shallow_model()
    input2=model(test_input)
    print(input2.size())
    input2=input2.view(N,2,10)
    input2=input2.view(N,20)
    print(input2.size())
    train_model2(model2,input2,train_target)
    nb_test_errors2 = compute_nb_errors(model, test_input, test_target)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    mean_error2[k]=nb_test_errors2


print(numpy.mean(mean_error)/test_input.size(0)*100)
######################################################################################################

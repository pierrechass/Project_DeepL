from utilities import *
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_model(model, train_input, train_target, test_input,test_target):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.05)
    nb_epochs = 50
    mini_batch_size = 200
    test_accuracy = torch.empty(nb_epochs).zero_()
    train_accuracy = torch.empty(nb_epochs).zero_()

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuracy[e] += utile.compute_nb_errors_standard(output, train_target.narrow(0, b, mini_batch_size))
        output_test = model(test_input)
        test_accuracy[e] = utile.compute_nb_errors(output_test, test_target)
    return test_accuracy,train_accuracy


def create_deep_model():
    return nn.Sequential(
        nn.Linear(2, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 2)
    )

utile = utilities(1000,1000)
train_input, train_target, test_input, test_targets = utile.create_datasets(plot=False)

mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

model = create_deep_model()
std = 1.

if std > 0:
    with torch.no_grad():
        for p in model.parameters(): p.uniform_(-1., 1.)


test_error, train_error = train_model(model, train_input.view(1000,-1), train_target.view(1000,-1), test_input.view(1000,-1), test_targets.view(1000,-1))

train_accuracy = train_error/1000 * 100
test_accuracy = test_error/1000 * 100

plt.plot(test_accuracy, label = "test")
plt.plot(train_accuracy, label = "train")
plt.legend()
plt.show()


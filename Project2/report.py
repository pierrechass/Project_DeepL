from utilities import utilities
import framework as frk
import torch
from matplotlib import pyplot as plt
import numpy as np
import statistics as st

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# As required, torch.grad is disabled
torch.set_grad_enabled(False)

# Define parameters
eta, batch_size = 0.1,100
epochs = 100
nb_laps = 50

# Creation of the model
model = frk.Sequential(frk.lossMSE(), frk.Linear(2,25), frk.tanh(), frk.Linear(25,25), frk.tanh(),
frk.Linear(25,25), frk.tanh(), frk.Linear(25,2), frk.tanh())

# Creation of the dataset
utile = utilities(1000,1000)
train_data, train_labels, test_data, test_labels = utile.create_datasets(plot=False)

mean, std = train_data.mean(), train_data.std()
train_data.sub_(mean).div_(std)
test_data.sub_(mean).div_(std)

# Containers for the datas
test_error = torch.empty(epochs).zero_()
train_error = torch.empty(epochs).zero_()
train_loss = torch.empty(epochs).zero_()
test_loss = torch.empty(epochs).zero_()

train_accuracy = np.empty((nb_laps,4))
test_accuracy = np.empty((nb_laps,4))

for j in range(nb_laps) :

    for i,eta in enumerate([1e-2, 5e-2, 1e-1, 5e-1]):
        model = frk.Sequential(frk.lossMSE(), frk.Linear(2,25), frk.tanh(), frk.Linear(25,25), frk.tanh(),
        frk.Linear(25,25), frk.tanh(), frk.Linear(25,2), frk.tanh())

        test_error = torch.empty(epochs).zero_()
        train_error = torch.empty(epochs).zero_()
        train_loss = torch.empty(epochs).zero_()
        test_loss = torch.empty(epochs).zero_()

        for epoch in range(epochs):
            for b in range(0,train_data.size(1),batch_size):

                # Forward pass
                loss, output = model.forward_pass(train_data.narrow(1,b,batch_size), train_labels.narrow(1,b,batch_size))
                # Update train loss
                train_loss[epoch] += loss
                # Backward pass and optimization
                model.backward_pass(output, train_labels.narrow(1,b,batch_size), eta=eta)
                # Update train error
                train_error[epoch] += utile.compute_nb_errors(output,train_labels.narrow(1,b,batch_size))
            
            # Update test error once the model is trained
            _, output_test = model.forward_pass(test_data, test_labels)
            test_error[epoch] = utile.compute_nb_errors(output_test, test_labels)


        train_accuracy[j,i] = train_error[-1]*100/test_data.size(1)
        test_accuracy[j,i] = test_error[-1]*100/test_data.size(1)

train_std = []
train_mean = []
test_std = []
test_mean = []
for n in range(4) :
    train_std.append(st.stdev(train_accuracy[:,n]))
    train_mean.append(st.mean(train_accuracy[:,n]))
    test_std.append(st.stdev(test_accuracy[:,n]))
    test_mean.append(st.mean(test_accuracy[:,n]))

fig, ax = plt.subplots()
labels = ['1e-2', '5e-2', '1e-1', '5e-1']
x = np.arange(len(labels)) 
width = 0.35
rects1 = ax.bar(x, test_mean,width, yerr = test_std)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.bar_label(rects1, padding=3)
ax.set_title("Mean and standard deviation over 50 runs of test accuracy by learning rate")
ax.set_ylabel("Final accuracy (%)")
ax.set_xlabel("eta")
plt.show()

# print("Test accuracy = %.2f" %test_accuracy[-1] + "%\n")
# print("Train accuracy = %.2f" %train_accuracy[-1]+ "%\n")

# plt.plot(test_accuracy, label = "test")
# plt.plot(train_accuracy, label = "train")
# plt.legend()
# plt.xlabel("epochs")
# plt.ylabel("Accuracy (%)")
# plt.title("Train and test accuracy during epochs : eta = %.2f, batch_size = %i\n" %(eta,batch_size)
#         + "Train accuracy = %.2f"%train_accuracy[-1] + "% " + "Test accuracy = %.2f" %test_accuracy[-1] + "%")
# plt.show()

# # denormalize datas and plot the output classification
# test_data.mul_(std).add_(mean)
# utile.plot_output(output_test,test_data)

# plt.plot(train_loss)
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.title("evolution of the MSE loss over the epochs")
# plt.show()


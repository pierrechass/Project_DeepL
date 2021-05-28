#Utilities 
"""
We have in this files all the functions that we will use to train or model and to test the performance of our model 
"""

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import time 
######################################################################
"""
Function to train our model :  
input : 
    model : the network we want to train 
    train_input, train_target, train_classes: the data and the labels to train our model
    train_input, train_target, train_classes: the data and the labels to test our model
    mini_batch_size : the batch size we use to train our model 
    nb_epochs : the number of epoch we use to train our model
    aux_loss : Select if we want to use auxiliary loss to train our model 
    weight : the weight use in the auxiliary loss 
    eta : the learning rate
    choose_optimizer : optimizer to use (SGD,Adam,Adadelta)
"""

def train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, error_train, error_test, nb_epochs = 25,aux_loss=True, weight=3, eta=1, choose_optimizer=2,plot=False):
    
    # reset weight :  
    #model.apply(weights_init)  get better results with the initialization by default

    model.train() # put in train mode (take into account dropout and batchnormalization)
    
    criterion = nn.CrossEntropyLoss()       # choice of the criterion for the loss

    #choice of the optimizer
    if choose_optimizer==0:
        optimizer = torch.optim.SGD(model.parameters(), lr = eta)

    elif choose_optimizer==1:
        optimizer = torch.optim.Adam(model.parameters(), lr = eta)
        print('Adam')
        print(eta)

    else :
        optimizer = torch.optim.Adadelta(model.parameters(), lr = eta) # weight_decay=0.0001


    t1=time.time() # use to compute the time computation of the training 
    
    for e in range(nb_epochs):
        acc_loss = 0
        model.train()   # put in train mode (take into account dropout and batchnormalization)
        for b in range(0, train_input.size(0), mini_batch_size):
            if model.__class__.__name__=='Netfinal': # different number of output depending on the model 
                class1,class2,output = model(train_input.narrow(0, b, mini_batch_size))

                # extract the true classes of each image (the first and second images of the pair)
                train_classes1=train_classes[:,0]
                train_classes2=train_classes[:,1]
            else :
                output = model(train_input.narrow(0, b, mini_batch_size))

            if aux_loss : 
                loss1 = criterion(output, train_target.narrow(0, b, mini_batch_size))   # Loss for the target
                loss2 = criterion(class1, train_classes1.narrow(0, b, mini_batch_size)) # Loss for the first image in the pair
                loss3 = criterion(class2, train_classes2.narrow(0, b, mini_batch_size)) # Loss for the second image in the pair
                loss  = loss1*weight+loss2+loss3          
                                             
            else : 
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
           
            acc_loss = acc_loss + loss.item()
            
            # Update our weight : 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Use to get the graph with the % of error on the training and testing set depending on the number of epochs 
        
        if plot:
            if model.__class__.__name__=='Netfinal':
                err_test,_,_=compute_nb_errors(model, test_input, test_target,test_classes)
                err_train,_,_=compute_nb_errors(model, train_input, train_target,train_classes)
                error_test[e]=err_test/test_input.size(0)*100
                error_train[e]=err_train/train_input.size(0)*100
            else:
                err_test=compute_nb_errors(model, test_input, test_target,test_classes)
                err_train=compute_nb_errors(model, train_input, train_target,train_classes)

                error_test[e]=err_test/test_input.size(0)*100
                error_train[e]=err_train/train_input.size(0)*100
    print('time train',time.time()-t1)
#######################################################################################################

"""
Function use to compute the number of errors in our model : 
Input : 
    model : Network that we want to test
    data_input, data_target, data_classes  : data and label of the data

Output : 
    if model = 'Netfinal'
        nb_data_errors_target : number of errors at the output (which image has the biggest number)
        nb_data_errors_class1 : number of error for image 1 of the pair (wrong number)
        nb_data_errors_class2 : number of error for image 2 of the pair (wrong number)
    else : 
        nb_data_errors_target : number of errors at the output (which image has the biggest number)
"""
def compute_nb_errors(model, data_input, data_target,data_classes):
    
    model.eval() # put in eval mode (no dropout and batchnormalization)

    nb_data_errors_target = 0
    nb_data_errors_classes1 = 0
    nb_data_errors_classes2 = 0


    # extract the predicted classes and target

    if model.__class__.__name__=='Netfinal': # different numbers of output depending on the model 
        x1,x2,output = model(data_input)
        _, predicted_target = torch.max(output, 1)
        _, predicted_classes1 = torch.max(x1, 1)
        _, predicted_classes2 = torch.max(x2, 1)

        # extract the true classes
        data_classes1=data_classes[:,0]
        data_classes2=data_classes[:,1]

        # compute the number of error 
        for k in range(data_target.size(0)):
            if data_target[k] != predicted_target[k]:  
                nb_data_errors_target = nb_data_errors_target + 1           # number of error for the input 
            if data_classes1[k] != predicted_classes1[k]:
                nb_data_errors_classes1 = nb_data_errors_classes1 + 1       # number of error for the classification of image 1 of the pair (number)
            if data_classes2[k] != predicted_classes2[k]:
                nb_data_errors_classes2 = nb_data_errors_classes2 + 1        # number of error for the classification of image 1 of the pair (number)
        return nb_data_errors_target,nb_data_errors_classes1,nb_data_errors_classes2
    
    else:
        output = model(data_input)
        _, predicted_target = torch.max(output, 1)
       
        # compute the number of error 
        for k in range(data_target.size(0)):
            if data_target[k] != predicted_target[k]:
                nb_data_errors_target = nb_data_errors_target + 1
        return nb_data_errors_target

######################################################################
"""
Function use to print our result and to stack results into different tensor
Input : 
    model : the network we want to train 
    train_input, train_target, train_classes: the data and the labels to train our model
    train_input, train_target, train_classes: the data and the labels to test our model
    mean_error_test : tensor to stack number of error for the test set (in %) at each run
    mean_error_train : tensor to stack number of error for the train set (in %) at each run
    k : the actual run
"""

def print_results(model,train_input, train_target, train_classes, test_input,test_target,test_classes,mean_error_test,mean_error_train, k,verbose=False):
    # compute number of errors in our testing set :

    if model.__class__.__name__=='Netfinal':
        nb_test_errors,error_class1, error_class2 = compute_nb_errors(model, test_input, test_target, test_classes)                              
        print('test error target {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
        if verbose : print('test error class1 {:0.2f}% {:d}/{:d}'.format((100 * error_class1) / test_input.size(0),
                                                      error_class1, test_input.size(0)))
        if verbose : print('test error class2 {:0.2f}% {:d}/{:d}'.format((100 * error_class2) / test_input.size(0),
                                                      error_class2, test_input.size(0)))
    else : 
        nb_test_errors = compute_nb_errors(model, test_input, test_target, test_classes)                              
        print('test error target {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    mean_error_test[k]=nb_test_errors/test_input.size(0)*100

    if model.__class__.__name__=='Netfinal':
        nb_train_errors,error_class1, error_class2 = compute_nb_errors(model, train_input, train_target, train_classes)                              
        print('train error target {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                        nb_train_errors, train_input.size(0)))
        if verbose : print('train error class1 {:0.2f}% {:d}/{:d}'.format((100 * error_class1) / train_input.size(0),
                                                        error_class1, train_input.size(0)))
        if verbose : print('train error class2 {:0.2f}% {:d}/{:d}'.format((100 * error_class2) / train_input.size(0),
                                                        error_class2, train_input.size(0)))
    else: 
        nb_train_errors = compute_nb_errors(model, train_input, train_target, train_classes)                              
        print('train error target {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                        nb_train_errors, train_input.size(0)))
    mean_error_train[k]=nb_train_errors/train_input.size(0)*100
######################################################################
"""
Function use to compute mean and standard deviation of the error (in %) for our test and training set
"""
def get_mean_std(error_test,error_train):
    print('mean test',error_test.mean())
    print('mean train',error_train.mean())

    print('std test',error_test.std())
    print('std train',error_train.std())
######################################################################

"""
Function use to initialise the weight 
"""
def weights_init(m):
    #torch.manual_seed(10)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

"""
Function use to plot the % or error of our training and testing set at each epoch
"""
def plot(test,train):
    fig, ax = plt.subplots()
    ax.plot(test,'r',label='test error')
    ax.plot(train,label='train error ')

    plt.xlabel('epochs')
    plt.ylabel('error [%]')
    leg = ax.legend()
    plt.show()
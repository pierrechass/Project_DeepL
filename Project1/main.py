# MAIN : 
import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import utilities as ut
import Models as M




# size of the mini_batch : 
mini_batch_size = 100
# number of epoch for our training :
epoch=25
# number of run we want make :
number_run=10
# MODEL we want to use : 
MODEL=3 #   1 : simple model 
        #   2 : complex model
        #   3 : resnet model

# initialize the tensor that stack the error at each round :  
error_test=torch.empty(number_run) # % of error once our model is trained at each run
error_train=torch.empty(number_run)


error_test_epoch=torch.empty(epoch)  #  % of error at each epoch (for one run)
error_train_epoch=torch.empty(epoch)

#error_test_epoch_nul=torch.empty(epoch)
#error_train_epoch_nul=torch.empty(epoch)
PLOT=False

for k in range(number_run):  # run 10 time our algorithm to get a better approximation of the performance of our model 
    # get the data : (already randomized in the function)
    N=1000
    [train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)
    
    # Normalized the data
    normalized=False
    if normalized:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    if MODEL==1:
        model= M.Netfinal(200, batchnorm=True, dropout=True, kernel_size=3,aux_loss_method=2) 
        ut.train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, error_train_epoch, error_test_epoch, aux_loss=True, weight=2, eta=1.1, choose_optimizer=2,plot=PLOT)  
        ut.print_results(model,train_input, train_target, train_classes, test_input,test_target,test_classes,error_test,error_train, k,verbose=False)    
    
    if MODEL==2:
        model= M.Netsimple(200)
        ut.train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, error_train_epoch, error_test_epoch, aux_loss=False, eta=0.03, choose_optimizer=2,plot=PLOT)  
        ut.print_results(model,train_input, train_target, train_classes, test_input,test_target,test_classes,error_test,error_train, k,verbose=False)    
    
    if MODEL==3:
        model= model3=M.ResNet(10,32) 
        ut.train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, error_train_epoch, error_test_epoch, aux_loss=False, eta=1, choose_optimizer=2,plot=PLOT)  
        ut.print_results(model,train_input, train_target, train_classes, test_input,test_target,test_classes,error_test,error_train, k,verbose=False)                  
                                                                       
ut.get_mean_std(error_test,error_train)

if PLOT:
    ut.plot(error_test_epoch,error_train_epoch)


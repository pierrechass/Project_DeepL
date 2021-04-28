from torch import empty
import math as m
import numpy as np


###################################
##        abstract class         ##
###################################

class Module ( object ) :
    def forward_pass ( self , * input ) :
        raise NotImplementedError
    def backward_pass ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return []


###################################
##      convolutional layer      ##
###################################

class Linear(Module) :

    def __init__(self, d_in, d_out, randType = 'normal', randArg1 = 0., randArg2 = 1.):

        # Define bias
        switch_b = {
            'normal' : empty(d_out).normal_(randArg1,randArg2),
            'uniform' : empty(d_out).uniform_(randArg1,randArg2)
        }
        self.b = switch_b.get(randType)

        # Define weights
        switch_w = {
            'normal' : empty((d_out, d_in)).normal_(randArg1,randArg2),
            'uniform' : empty((d_out, d_in)).uniform_(randArg1,randArg2)
        }
        self.w = switch_w.get(randType)

        if (self.w is None) : print("Error : wrong randType value for init of fully connected layer")

        # Define input
        self.x = empty(d_in).zero_()

        # Define gradients
        self.dl_dw = empty((d_out,d_in)).zero_()
        self.dl_db = empty(d_out).zero_()

    def forward_pass(self, input):
        self.x = input
        return self.w @ self.x + self.b
    
    def backward_pass(self, dl_ds):
        self.dl_db.add_(dl_ds)
        self.dl_dw.add_(self.dl_db.view(-1, 1).mm(self.x.view(1, -1)))
        
        return self.w.t()@dl_ds

    def gradient_step(self, eta):
        self.w = self.w - eta*self.dl_dw
        self.b = self.b - eta*self.dl_db
    
    def reset_gradients(self):
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def param(self):
        return [(self.w, self.b), (self.dl_dw, self.dl_db)]
        

###################################
##     tanh activation layer     ##
###################################

class tanh(Module):

    def __init__(self):
        self.s = None

    def forward_pass(self, input):
        self.s = input
        return self.s
    
    def backward_pass(self, dl_dx):
        dtanh = 4 * (self.s.exp() + self.s.mul(-1).exp()).pow(-2)
        return dtanh * dl_dx

class reLU(Module):
    def __init__(self):
        self.s = None
    
    def forward_pass(self, input):
        self.s = input
        return torch.clamp(self.s,min=0)
    
    def backward_pass(self,x,dl_dx):
        dreLU = float(self.s>0)
        return dreLU * dl_dx

###################################
##             loss              ##
###################################

class MSE(Module):
    
    def forward_pass(self, input, target):
        return (input - target).pow(2).sum()/input.size(0)

    def backward_pass(self, input, target):
        return 2 * (input - target)
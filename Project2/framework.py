from torch import empty
import torch
import math as m


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

        # Define bia
        self.b = empty(d_out,1).zero_()

        # Define weights
        switch_w = {
            'normal' : empty((d_out, d_in)).normal_(randArg1,randArg2).mul_(1/m.sqrt(d_in)),
            'uniform' : empty((d_out, d_in)).uniform_(-1/m.sqrt(d_in),1/m.sqrt(d_in))
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
        self.dl_db = dl_ds.sum(1).view(-1,1)
        self.dl_dw = (self.x @ dl_ds.t()).t()

        return self.w.t() @ dl_ds

    def gradient_step(self, eta):
        self.w = self.w - eta*self.dl_dw
        self.b = self.b - eta*self.dl_db
    
    def reset_gradients(self):
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def param(self):
        return [(self.w, self.b), (self.dl_dw, self.dl_db)]
        

###################################
##       activation layers       ##
###################################

class tanh(Module):

    def __init__(self):
        self.s = None

    def forward_pass(self, input):
        self.s = input
        return self.s.tanh()
    
    def backward_pass(self, dl_dx):
        dtanh = 4 * (self.s.exp() + self.s.mul(-1).exp()).pow(-2)
        return dtanh * dl_dx
    
    def param(self):
        return [self.s]

class reLU(Module):
    def __init__(self):
        self.s = None
    
    def forward_pass(self, input):
        self.s = input
        return torch.clamp(self.s,min=0)
    
    def backward_pass(self ,dl_dx):
        dreLU = self.s>0
        return dreLU.float() * dl_dx

    def param(self):
        return [self.s]


###################################
##             loss              ##
###################################

class lossMSE(Module):
    
    def forward_pass(self, input, target):
        return (input - target).pow(2).sum()/input.size(1)

    def backward_pass(self, input, target):
        return 2 * (input - target)/input.size(1)
    
    def param(self):
        return []

###################################
##          Sequential           ##
###################################

class Sequential(Module):
    def __init__(self, loss, *kargs):
        self.layers = list(kargs)
        if isinstance(self.layers[-1],lossMSE):
            raise TypeError("Last element should not be a loss : loss already specified")
        self.layers.append(loss)
    
    def __str__(self):
        out = 'Model containing: \n'
        for i, layer in enumerate(self.layers[:-2]):
            out = out + ('{} : '.format(i+1)) + str(layer) + '\n'
        out = out + 'With loss : \n' + str(self.layers[-1])
        return out

    def forward_pass(self, input, target):
        next = self.layers[0].forward_pass(input)
        for layer in self.layers[1:-1]:
            next = layer.forward_pass(next)
        loss = self.layers[-1].forward_pass(next,target)
        return loss,next
    
    def backward_pass(self, input, target, eta = 0.01):
        out = self.layers[-1].backward_pass(input,target)
        for layer in reversed(self.layers[:-1]):
            out = layer.backward_pass(out)
            if isinstance(layer,Linear) :
                layer.gradient_step(eta)
        return out
    
    def param(self):
        out = [layer.param() for layer in self.layers]
        return out
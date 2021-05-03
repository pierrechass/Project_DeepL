import torch
from matplotlib import pyplot as plt
import math as m

class utilities():

    def __init__(self, size = 1000):
        self.Size = size
    
    def create_datasets(self,plot = False):

        datas = torch.empty(self.Size,2).uniform_()
        labels = torch.zeros(self.Size)
        labels[((datas[:,0] - 0.5)**2 + (datas[:,1]-0.5)**2 <= 1/(2*m.pi))] = 1

        if plot ==True :
            plt.scatter(datas[labels == 0,0],datas[labels == 0,1], label = 'class zero')
            plt.scatter(datas[labels == 1,0],datas[labels == 1,1], label = 'class one')
            plt.set_title("ing dataset with %i datapoints" %self.Size)
            plt.legend()

        return datas

            

        
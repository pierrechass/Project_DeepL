import torch
from matplotlib import pyplot as plt
import math as m

class utilities():

    def __init__(self, testSize = 1000, trainSize = 1000):
        self.testSize = testSize
        self.trainSize = trainSize
    
    def create_datasets(self,plot = False):

        train_datas = torch.empty(2,self.trainSize).uniform_()
        train_labels = torch.zeros(2,self.trainSize)
        # One hot encoding
        for i in range(self.trainSize):
            if ((train_datas[0,i] - 0.5)**2 + (train_datas[1,i]-0.5)**2 <= 1/(2*m.pi)):
                train_labels[1,i] = 1.
            else :
                train_labels[0,i] = 1.

        test_datas = torch.empty(2,self.testSize).uniform_()
        test_labels = torch.zeros(2,self.testSize)
        # One hot encoding
        for i in range(self.testSize):
            if ((test_datas[0,i] - 0.5)**2 + (test_datas[1,i]-0.5)**2 <= 1/(2*m.pi)):
                test_labels[1,i] = 1.
            else :
                test_labels[0,i] = 1.

        if plot ==True :
            fig, axs = plt.subplots(1,2, figsize=(4,4))
            axs[0].scatter(train_datas[0,train_labels[0,:] == 1],train_datas[1,train_labels[0,:] == 1], label = 'class zero')
            axs[0].scatter(train_datas[0,train_labels[1,:] == 1],train_datas[1,train_labels[1,:] == 1], label = 'class one')
            axs[0].set_title("Training dataset with %i datapoints" %self.trainSize)
            axs[0].legend()

            axs[1].scatter(test_datas[0,test_labels[0,:] == 1],test_datas[1,test_labels[0,:] == 1], label = 'class zero')
            axs[1].scatter(test_datas[0,test_labels[1,:] == 1],test_datas[1,test_labels[1,:] == 1], label = 'class one')
            axs[1].set_title("Testing dataset with %i datapoints" %self.testSize)
            axs[1].legend()

            plt.show()

        return train_datas, train_labels, test_datas, test_labels

    def compute_nb_errors(self, model_out, labels):
        nb_errors = 0
        _,predicted = torch.max(model_out,0)
        _,target = torch.max(labels,0)
        for i in range(predicted.size(0)):
            if predicted[i] != target[i] :
                nb_errors += 1
        
        return nb_errors

    def compute_nb_errors_standard(self, model_out, labels):
        nb_errors = 0
        _,predicted = torch.max(model_out,1)
        _,target = torch.max(labels,1)
        for i in range(predicted.size(0)):
            if predicted[i] != target[i] :
                nb_errors += 1
        
        return nb_errors
    
    def plot_output(self,output,test_data):

        plt.figure(figsize=(6,6))
        _,predicted = torch.max(output,0)
        plt.scatter(test_data[0,predicted == 0],test_data[1,predicted == 0], label = "Class zero")
        plt.scatter(test_data[0,predicted== 1],test_data[1,predicted == 1], label = "Class one")

        circle = plt.Circle((0.5, 0.5), 1/m.sqrt(2*m.pi), fill=False)
        ax = plt.gca()
        ax.add_patch(circle)

        plt.show()


            
            


            

        
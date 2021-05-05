# Projet 1

import torch
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy

N=1000
[train_input, train_target, train_classes, test_input, test_target, test_classes]=  prologue.generate_pair_sets(N)

dropout = nn.Dropout()
model = nn.Sequential(nn.Linear(3, 10), dropout, nn.Linear(10, 3))
dropout.test(False)

print(dropout.training)


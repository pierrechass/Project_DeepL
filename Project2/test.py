from utilities import utilities
import framework as frk
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# lin = framework.Linear(4,1, randType = 'normal', randArg1 = 0., randArg2 = 1.)
# sigma = framework.tanh()
# loss = framework.MSE()

# util = utilities()

# inp = torch.ones([4])

# inp2 = torch.zeros(4)
# inp2[0] = 1.

# target = 1.

# train = util.create_datasets(plot=True)

# for i in range(30):

#     #lin.reset_gradients()

#     print("param = ",lin.param())
#     out = lin.forward_pass(inp2)
#     out = sigma.forward_pass(out)
#     loss_act = loss.forward_pass(out,target)

#     dl_dx = loss.backward_pass(out,target)
#     dl_ds = sigma.backward_pass(dl_dx)
#     lin.backward_pass(dl_ds)

#     lin.gradient_step(0.015)

#     print(out)
#     print(loss_act)

model = frk.Sequential(frk.lossMSE(), frk.Linear(10,10), frk.reLU(), frk.Linear(10,5), frk.reLU())
print(model.param())


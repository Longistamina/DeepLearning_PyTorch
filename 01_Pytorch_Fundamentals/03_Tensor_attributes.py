'''
1. tensor.ndim: the number of dimensions (rank) the tensor has.

2. tensor.shape: the size of each dimension in a torch.Size object.

3. tensor.dtype: the data type of the tensor

4. tensor.device: the device the tensor is on

5. tensor.requires_grad: the boolean indicating whether PyTorch should track gradients for this tensor
'''

import torch

tensor_cpu = torch.tensor(
    [[2, 5, 8, 9], 
     [1, 5, 3, 4]], 
    dtype=torch.int8, 
    device="cpu", 
    requires_grad=False
)

torch.manual_seed(0)
tensor_cuda = torch.randn(size=(3, 3, 5), dtype=torch.half, device="cuda", requires_grad=True)
print(tensor_cuda)
# tensor([[[-0.9248, -0.4253, -2.6445,  0.1451, -0.1208],
#          [-0.5796, -0.6230, -0.3284, -1.0742, -0.3630],
#          [-1.6709,  2.2656,  0.3118, -0.1842,  1.2871]],

#         [[ 1.1816, -0.1271,  1.2168,  1.4355,  1.0605],
#          [-0.4941, -1.4248, -0.7246, -1.2969,  0.0697],
#          [-0.0074,  1.8965,  0.6880, -0.0779, -0.8374]],

#         [[ 1.3506, -0.2878, -0.5967, -0.3284, -0.9087],
#          [-0.8062, -0.7407, -0.0504,  0.5435,  1.5146],
#          [ 0.0141,  0.4531,  1.6348,  0.7124, -0.1805]]], device='cuda:0',
#        dtype=torch.float16)


#################
## tensor.ndim ##
#################

print(tensor_cpu.ndim)
# 2

print(tensor_cuda.ndim)
# 3

##################
## tensor.shape ##
##################

print(tensor_cpu.shape)
# torch.Size([2, 4])

print(tensor_cuda.shape)
# torch.Size([3, 3, 5])

##################
## tensor.dtype ##
##################

print(tensor_cpu.dtype)
# torch.int8

print(tensor_cuda.dtype)
# torch.float16

###################
## tensor.device ##
###################

print(tensor_cpu.device)
# cpu

print(tensor_cuda.device)
# cuda:0
# (cuda:0 means this tensor is on the first gpu)

##########################
## tensor.requires_grad ##
##########################

print(tensor_cpu.requires_grad)
# False

print(tensor_cuda.requires_grad)
# True
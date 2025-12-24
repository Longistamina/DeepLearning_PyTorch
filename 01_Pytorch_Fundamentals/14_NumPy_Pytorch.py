'''
1. From NumPy to Pytorch: torch.from_numpy(arr)
   (float64 -> float64)

2. From Pytorch to NumPy: tensor.numpy()
   (float32 -> float32)
'''

import torch
import numpy as np

tensor = torch.tensor([2, 3, 4.5])
print(tensor.dtype)
# torch.float32

arr = np.array([5.1, 6, 8.2, 9])
print(arr.dtype)
# float64

###########################
## torch.from_numpy(arr) ##
###########################

tensor_fromArr = torch.from_numpy(arr)

print(tensor_fromArr)
# tensor([5.10, 6.00, 8.20, 9.00], dtype=torch.float64)

print(tensor_fromArr.dtype)
# torch.float64

'''
The default dtype in NumPy for float numbers is float64.

So when using torch.from_numpy() to convert it into torch,
the resulted tensor will be defined as torch.float64
'''

####################
## tensor.numpy() ##
####################

arr_fromTensor = tensor.numpy()

print(arr_fromTensor)
# [2.  3.  4.5]

print(arr_fromTensor.dtype)
# float32

'''
The default dtype in torch for float numbers is torch.float32.

So when using tensor.numpy()) to convert it into a numpy array,
the resulted array will be defined as float32.
'''

tensor_gpu = torch.tensor([1, 2, 3], device="cuda")
print(tensor_gpu.numpy())
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
'''NOTE: if a tensor is on GPU ("cuda"), then tensor.numpy() will return ERROR'''

# First, bring it back to .cpu(), then .numpy()
tensor_cpu_numpy = tensor_gpu.cpu().numpy()
print(tensor_cpu_numpy)
# [1 2 3]
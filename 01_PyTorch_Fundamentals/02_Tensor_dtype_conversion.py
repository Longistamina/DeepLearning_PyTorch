'''
1. torch.tensor(dtype=...)

2. tensor.type(dtype): dtype conversion
'''

import torch


#-----------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. torch.tensor(dtype=...) --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#
'''
The default dtype of tensor is torch.float32

We can specify other type like torch.float16 or torch.float64
(The higher the more precise floating-point numbers: https://en.wikipedia.org/wiki/Precision_(computer_science))

All the dtypes supported: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
'''

tensor_None = torch.tensor([3, 2.5, 6, 8.9], dtype=None)
print(tensor_None.dtype)
# torch.float32 (default, even if we set it as None)

tensor_16 = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float16)
print(tensor_16.dtype)
# torch.float16

tensor_64 = torch.tensor([1, 2, 3, 5, 7], dtype=torch.float64)
print(tensor_64.dtype)
# torch.float64


#------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. tensor.type(dtype) --------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

tensor_original = torch.tensor([25.2, 3.7, 8.5, 6.329])
print(tensor_original.dtype)
# torch.float32

tensor_half = tensor_original.type(torch.half)
print(tensor_half.dtype)
# torch.float16

tensor_int64 = tensor_original.type(torch.int64)
print(tensor_int64) # tensor([25,  3,  8,  6]) (Not rounding up for 3.7 and 8.5)
print(tensor_int64.dtype) # torch.int64



'''
1. torch.tensor(dtype=...)

2. dtype conversion
'''

import torch


#-----------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. torch.tensor(dtype=...) --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#
'''
The default dtype of tensor is torch.float32

We can specify other type like torch.float16 or torch.float64
(The higher the more precise floating-point numbers)
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
'''
1. Squeeze: torch.squeeze(input, dim=None), tensor.squeeze(dim=None)
   + Removes dimensions of size 1 from the tensor.
   + If dim is specified, it only removes that specific dimension if its size is 1.

2. Unsqueeze: torch.unsqueeze(input, dim), tensor.unsqueeze(dim)
   + Inserts a dimension of size 1 at the specified position.
   + The 'dim' argument is required.
'''

import torch

tensor_A = torch.zeros(1, 3, 1, 4)
print(f"Original shape: {tensor_A.shape}")
# Original shape: torch.Size([1, 3, 1, 4])

print(tensor_A)
# tensor([[[[0., 0., 0., 0.]],

#          [[0., 0., 0., 0.]],

#          [[0., 0., 0., 0.]]]])

#-----------------------------------------------------------------------------------------------------------#
#--------------------------------------------- 1. Squeeze --------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
.squeeze(): Removes ALL dimensions of size 1.
.squeeze(dim=n): Removes dimension n ONLY if its size is 1.
'''

########################
## Squeeze All (None) ##
########################

squeezed_all = tensor_A.squeeze()

print(squeezed_all)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

print(squeezed_all.shape)
# torch.Size([3, 4]) 
# Both dimensions at index 0 and 2 (size 1) were removed.

###########################
## Squeeze Specific Dim  ##
###########################

# Squeeze the first dimension (index 0) which is size 1
print(tensor_A.squeeze(dim=0).shape)
# torch.Size([3, 1, 4])

# Squeeze the third dimension (index 2) which is size 1
print(tensor_A.squeeze(dim=2).shape)
# torch.Size([1, 3, 4])

# Attempt to squeeze a dimension that is NOT size 1 (index 1 is size 3)
print(tensor_A.squeeze(dim=1).shape)
# torch.Size([1, 3, 1, 4])
# Nothing happens because the size is not 1.


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------------- 2. Unsqueeze -------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
.unsqueeze(dim=n): Adds a new dimension of size 1 at the index n.
The resulting tensor has one more dimension than the original.
'''

tensor_v = torch.arange(1., 4.) # Shape: [3]

print(tensor_v)
# tensor([1., 2., 3.])

print(f"Vector shape: {tensor_v.shape}")
# Vector shape: torch.Size([3])

########################
## Unsqueeze at Start ##
########################

# Add dimension at index 0 (turns vector into row-like matrix)
print(tensor_v.unsqueeze(dim=0))
# tensor([[1., 2., 3.]])
# torch.Size([1, 3])

print(tensor_v.unsqueeze(dim=1))
# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])

#########################
## Unsqueeze in Middle ##
#########################

tensor_M = torch.randn(2, 3) # Shape: [2, 3]
print(tensor_M)
# tensor([[ 0.27, -0.92, -0.04],
#         [ 0.29, -0.01, -0.91]])

print(tensor_M.unsqueeze(dim=1))
# tensor([[[ 0.27, -0.92, -0.04]],

#         [[ 0.29, -0.01, -0.91]]])
# torch.Size([2, 1, 3])

######################
## Unsqueeze at End ##
######################

# Using -1 to target the last possible position
print(tensor_v.unsqueeze(dim=-1))
# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])
# Effectively turns a 1D vector into a column vector.

# Common use case: Preparing a single image for a CNN that expects [Batch, Channel, H, W]
image = torch.randn(3, 224, 224) # [C, H, W]
batch_ready = image.unsqueeze(dim=0)
print(batch_ready.shape)
# torch.Size([1, 3, 224, 224])
'''
1. Basic Indexing & Slicing: tensor[i, j], tensor[start:end:step]
   + Accessing elements, rows, and columns
   + Using Ellipsis (...)

2. Boolean Indexing (Masking): tensor[condition]
   + Filtering data based on logic

3. Advanced Selection Functions:
   + torch.index_select(input, dim, index)
   + torch.masked_select(input, mask)
   + torch.gather(input, dim, index)
'''

import torch
torch.set_printoptions(precision=2)

# Create sample tensors
tensor_v = torch.arange(10, 19) 
# tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])

tensor_M = torch.tensor([[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9],
                         [10, 11, 12]])
# Shape: (4, 3)

#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 1. Basic Indexing & Slicing ----------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
Indexing in PyTorch follows NumPy conventions.
Slicing [start:end:step] is non-inclusive of the 'end' index.
'''

#################
## With Vector ##
#################

print(tensor_v[0])      # First element: 10
print(tensor_v[-1])     # Last element: 18
print(tensor_v[2:5])    # Slice: tensor([12, 13, 14])

#################
## With Matrix ##
#################

# Get row 1 (index 1)
print(tensor_M[1, :])   
# tensor([4, 5, 6])

# Get column 2 (index 2)
print(tensor_M[:, 2])   
# tensor([3, 6, 9, 12])

# Get a sub-matrix (middle 2 rows, first 2 columns)
print(tensor_M[1:3, 0:2])
# tensor([[4, 5],
#         [7, 8]])

# Get the last element of 1-index column (2nd column)
print(tensor_M[:, 1][-1])
# tensor(11)

#################
##  Ellipsis   ##
#################
'''
The Ellipsis (...) is used to represent "all other dimensions".
Useful for high-dimensional tensors.
'''
tensor_4D = torch.randn(2, 3, 4, 5)

# Select all elements in first 3 dims, but only index 0 of the last dim
print(tensor_4D[..., 0].shape) 
# torch.Size([2, 3, 4])


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 2. Boolean Indexing (Masking) --------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
Returns a 1D tensor containing only elements that satisfy the condition.
'''

# Find all elements in M greater than 7
mask = tensor_M > 7
print(mask)
# tensor([[False, False, False],
#         [False, False, False],
#         [False,  True,  True],
#         [ True,  True,  True]])

print(tensor_M[mask])
# tensor([8, 9, 10, 11, 12])


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------- 3. Advanced Selection Functions ---------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

########################
## torch.index_select ##
########################
'''
Selects specific indices along a dimension. 
The index must be a LongTensor.
'''

indices = torch.tensor([0, 3])
# Select row 0 and row 3

print(torch.index_select(tensor_M, dim=0, index=indices))
# tensor([[ 1,  2,  3],
#         [10, 11, 12]])

#########################
## torch.masked_select ##
#########################
'''
Functional version of boolean indexing.
'''

result = torch.masked_select(tensor_M, tensor_M.lt(5)) # .lt is "less than"
print(result)
# tensor([1, 2, 3, 4])

####################
##  torch.gather  ##
####################
'''
Gathers values along an axis specified by dim using an index map.
Useful for selecting specific class probabilities in NLP or RL.
'''

data = torch.tensor([[10, 20], 
                     [30, 40]])
# For row 0, take col 1. For row 1, take col 0.
idx = torch.tensor([[1], [0]])

gathered = torch.gather(data, dim=1, index=idx)
print(gathered)
# tensor([[20],
#         [30]])
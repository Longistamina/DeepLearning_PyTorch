'''
1. Stack: torch.stack(tensors, dim=0)
   + Concatenates a sequence of tensors along a NEW dimension.

2. Vertical Stack: torch.vstack(tensors) / torch.row_stack(tensors)
   + Stacks tensors vertically (row-wise).
   
3. Horizontal Stack: torch.hstack(tensors)
   + Stacks tensors horizontally (column-wise).

4. Column Stack: torch.column_stack(tensors)
   + Specifically stacks 1D tensors as columns in a 2D result.

5. Depth Stack: torch.dstack(tensors)
   + Stacks tensors along the third dimension (depth).
'''

import torch

# Create sample 1D vectors
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

# Create sample 2D matrices
m1 = torch.tensor([[1, 1], 
                   [1, 1]])

m2 = torch.tensor([[2, 2], 
                   [2, 2]])

#-----------------------------------------------------------------------------------------------------------#
#--------------------------------------------- 1. Stack ----------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.stack() joins tensors along a NEW dimension. 
All tensors must be the same size.
'''

print(torch.stack((t1, t2), dim=0))
# tensor([[1, 2, 3],
#         [4, 5, 6]]) 
# Result shape: [2, 3]

print(torch.stack((t1, t2), dim=1))
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# Result shape: [3, 2]


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 2. Vertical & Row Stack --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.vstack() stacks tensors on top of each other.
torch.row_stack() is an alias for vstack.
- 1D tensors (size N) become a matrix (2 x N).
- 2D tensors (M x N) become a larger matrix ((M+M) x N).
'''

print(torch.vstack((t1, t2)))
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(torch.row_stack((m1, m2)))
# tensor([[1, 1],
#         [1, 1],
#         [2, 2],
#         [2, 2]])


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------- 3. Horizontal Stack -------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.hstack() stacks tensors side-by-side.
- 1D tensors (size N) are concatenated into a longer 1D tensor (size 2N).
- 2D tensors (M x N) become a wider matrix (M x (N+N)).
'''

print(torch.hstack((t1, t2)))
# tensor([1, 2, 3, 4, 5, 6])

print(torch.hstack((m1, m2)))
# tensor([[1, 1, 2, 2],
#         [1, 1, 2, 2]])


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 4. Column Stack ----------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.column_stack() is similar to hstack, but it behaves differently for 1D tensors.
It ensures 1D tensors are treated as COLUMNS (vertical) of a 2D matrix.
'''

print(torch.column_stack((t1, t2)))
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# Result shape: [3, 2] (Compare this to hstack which stayed 1D)


#-----------------------------------------------------------------------------------------------------------#
#---------------------------------------- 5. Depth Stack ---------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.dstack() stacks tensors along the third dimension (depth).
- 1D tensors (N) become (1 x N x 2).
- 2D tensors (M x N) become (M x N x 2).
'''

print(torch.dstack((t1, t2)))
# tensor([[[1, 4],
#          [2, 5],
#          [3, 6]]])
# Result shape: [1, 3, 2]

print(torch.dstack((m1, m2)))
# tensor([[[1, 2],
#          [1, 2]],

#         [[1, 2],
#          [1, 2]]])
# Result shape: [2, 2, 2]
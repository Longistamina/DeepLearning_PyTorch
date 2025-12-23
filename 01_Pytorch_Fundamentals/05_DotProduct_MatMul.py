'''
1. Dot Product of two 1D tensors with SAME SIZE 
   (n, ) @ (n, ) = scalar (1, )
   + torch.dot(tensor_v1, tensor_v2)
   + tensor_v1.dot(tensor_v2)
   + tensor_v1 @ tensor_v2 (not recommended)

2. Matrix Product of two 2D tensors with SAME INNER SIZE 
   (n, m) @ (m, p) = (n, p)
   + torch.matmul(tensor_M1, tensor_M2)
   + tensor_M1.matmul(tensor_M2)
   + tensor_M1 @ tensor_M2
   
https://en.wikipedia.org/wiki/Dot_product
'''

import torch


#-----------------------------------------------------------------------------------------------------------#
#--------------------------- 1. Dot Product of two 1D tensors with SAME SIZE -------------------------------#
#-----------------------------------------------------------------------------------------------------------#


tensor_v1 = torch.tensor([1, 3, 5, 7]) # (4,)
tensor_v2 = torch.tensor([2, 4, 6, 8]) # (4, )

tensor_v3 = torch.tensor([44, 53])     # (2, )

#######################
## torch.dot(v1, v2) ##
#######################

print(torch.dot(tensor_v1, tensor_v2))
# tensor(100)

print(torch.dot(tensor_v1, tensor_v3))
'''RuntimeError: inconsistent tensor size, expected tensor [4] and src [2] to have the same number of elements, but got 4 and 2 elements respectively'''

##############################
## tensor_v1.dot(tensor_v2) ##
##############################

print(tensor_v2.dot(tensor_v1))
# tensor(100)

print(tensor_v2.dot(tensor_v3))
'''RuntimeError: inconsistent tensor size, expected tensor [4] and src [2] to have the same number of elements, but got 4 and 2 elements respectively'''

#############################################
## tensor_v1 @ tensor_v2 (Not recommended) ##
#############################################
'''In Python, @ is for dot product, but not recommend to use since it's slower than using torch method'''

print(tensor_v1 @ tensor_v2)
# tensor(100)

print(tensor_v2 @ tensor_v3)
'''RuntimeError: inconsistent tensor size, expected tensor [4] and src [2] to have the same number of elements, but got 4 and 2 elements respectively'''


#--------------------------------------------------------------------------------------------------------------------#
#--------------------------- 2. Matrix Product of two 2D tensors with SAME INNER SIZE -------------------------------#
#--------------------------------------------------------------------------------------------------------------------#


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

torch.manual_seed(42)
tensor_M1 = torch.randint(1, 11, size=(2, 3), dtype=torch.float32) # Set dtype to float32 for dtype synchronization
print(tensor_M1)
# tensor([[3., 8., 7.],
#         [5., 7., 6.]])

torch.manual_seed(42)
tensor_M2 = torch.randn(size=(3, 5))
print(tensor_M2)
# tensor([[ 0.3367,  0.1288,  0.2345,  0.2303, -1.1229],
#         [-0.1863,  2.2082, -0.6380,  0.4617,  0.2674],
#         [ 0.5349,  0.8094,  1.1103, -1.6898, -0.9890]])

########################################
## torch.matmul(tensor_M1, tensor_M2) ##
########################################
'''Same as torch.mm(tensor_M1, tensor_M2)'''

print(torch.matmul(tensor_M1, tensor_M2))
# tensor([[ 3.2638, 23.7175,  3.3714, -7.4443, -8.1525],
#         [ 3.5886, 20.9576,  3.3681, -5.7555, -9.6766]])

print(torch.matmul(tensor_M1, tensor_M2).shape)
# torch.Size([2, 5])
# 2x3 @ 3x5 = 2x5

print(torch.matmul(tensor_M2, tensor_M1))
'''RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x5 and 2x3)'''

#################################
## tensor_M1.matmul(tensor_M2) ##
#################################
'''Same as tensor_M1.mm(tensor_M2)'''

print(tensor_M1.matmul(tensor_M2))
# tensor([[ 3.2638, 23.7175,  3.3714, -7.4443, -8.1525],
#         [ 3.5886, 20.9576,  3.3681, -5.7555, -9.6766]])

print(tensor_M1.matmul(tensor_M2).shape)
# torch.Size([2, 5])
# 2x3 @ 3x5 = 2x5

print(tensor_M2.matmul(tensor_M1))
'''RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x5 and 2x3)'''

#############################################
## tensor_M1 @ tensor_M2 (Not recommended) ##
#############################################

print(tensor_M1 @ tensor_M2)
# tensor([[ 3.2638, 23.7175,  3.3714, -7.4443, -8.1525],
#         [ 3.5886, 20.9576,  3.3681, -5.7555, -9.6766]])

print((tensor_M1 @ tensor_M2).shape)
# torch.Size([2, 5])
# 2x3 @ 3x5 = 2x5

print(tensor_M2 @ tensor_M1)
'''RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x5 and 2x3)'''

'''
1. Trigonometric: sin(), cos(), tan()
   + Vector vs Matrix application

2. Exponential & Logarithmic: exp(), log(), sqrt()
   + Vector vs Matrix application

3. Power & Absolute: pow(), abs()
   + Vector vs Matrix application

4. Rounding & Truncation: floor(), ceil(), round()
   + Vector vs Matrix application
   
5. Activation-like Ops: sigmoid(), tanh(), relu()

Note: Most functions have an in-place version (e.g., .sin_()) that modifies 
the tensor directly to save memory.
'''

import torch
torch.set_printoptions(precision=4)

# Create sample 1D vector
tensor_v = torch.tensor([-1.0, 0.0, 1.0])

# Create sample 2D matrix (2x2)
tensor_M = torch.tensor([[-1.0, 0.5],
                         [ 1.2, 2.0]])

#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Trigonometric Functions --------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
Applies the function to every element individually.
'''

#################
## With Vector ##
#################
print(torch.sin(tensor_v))
# tensor([-0.8415,  0.0000,  0.8415])

#################
## With Matrix ##
#################
print(torch.cos(tensor_M))
# tensor([[ 0.5403,  0.8776],
#         [ 0.3624, -0.4161]])


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 2. Exponential & Logarithmic ---------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

#################
## With Vector ##
#################
print(torch.exp(tensor_v))
# tensor([0.3679, 1.0000, 2.7183])

#################
## With Matrix ##
#################
# Note: log() will return 'nan' for negative numbers
print(torch.sqrt(tensor_M.abs())) 
# tensor([[1.0000, 0.7071],
#         [1.0954, 1.4142]])


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Power & Absolute Value ---------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

#################
## With Vector ##
#################
print(torch.pow(tensor_v, 2))
# tensor([1., 0., 1.])

#################
## With Matrix ##
#################
print(torch.abs(tensor_M))
# tensor([[1.0000, 0.5000],
#         [1.2000, 2.0000]])


#-----------------------------------------------------------------------------------------------------------#
#--------------------------------------- 4. Rounding & Truncation ------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

#################
## With Vector ##
#################
tensor_v_float = torch.tensor([1.2, 1.8, -1.5])
print(torch.round(tensor_v_float))
# tensor([ 1.,  2., -2.])

#################
## With Matrix ##
#################
tensor_M_float = torch.tensor([[1.1, 2.9],
                               [-0.4, 4.5]])

print(torch.floor(tensor_M_float))
# tensor([[ 1.,  2.],
#         [-1.,  4.]])

print(torch.ceil(tensor_M_float))
# tensor([[ 2.,  3.],
#         [-0.,  5.]])

#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 5. Activation-like Ops ------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
While these are often in torch.nn.functional, they exist as 
vectorized math ops in the main torch module as well.
'''

print(torch.sigmoid(tensor_v))
# tensor([0.2689, 0.5000, 0.7311, 0.8808])

print(torch.tanh(tensor_v))
# tensor([-0.7616,  0.0000,  0.7616,  0.9640])

print(torch.relu(tensor_v))
# tensor([0., 0., 1., 2.])
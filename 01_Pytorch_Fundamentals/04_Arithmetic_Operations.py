'''
1. Addition (+)
2. Substraction (-)
3. Multiplication (*): element-wise
4. Division (/)
5. Floor division (//)
6. Remainder (%)
7. Exponential (**)
8. Reassign to update values
'''

import torch

torch.manual_seed(0)

tensor_vector = torch.randint(low=1, high=21, size=(5,))
print(tensor_vector)
# tensor([ 5, 20, 14,  1,  4])

tensor_matrix = torch.randn(size=(3, 2)).round(decimals=2)
print(tensor_matrix)
# tensor([[ 0.5500,  0.2700],
#         [ 0.6500,  0.2500],
#         [-0.3400,  0.4600]])


##################
## Addition (+) ##
##################
'''Same as tensor.add(val)'''

print(tensor_vector + 10)
# tensor([15, 30, 24, 11, 14])

print(tensor_matrix + 2.5)
# tensor([[3.0500, 2.7700],
#         [3.1500, 2.7500],
#         [2.1600, 2.9600]])

#####################
## Subtraction (-) ##
#####################
'''Same as tensor.subtract(val)'''

print(tensor_vector - 5)
# tensor([ 0, 15,  9, -4, -1])

print(tensor_matrix - 3.2)
# tensor([[-2.6500, -2.9300],
#         [-2.5500, -2.9500],
#         [-3.5400, -2.7400]])

########################
## Multiplication (*) ##
########################
'''
Same as tensor.multiply(val)
element-wise multiplication
'''

print(tensor_vector * 2)
# tensor([10, 40, 28,  2,  8])

print(tensor_matrix * 3)
# tensor([[ 1.6500,  0.8100],
#         [ 1.9500,  0.7500],
#         [-1.0200,  1.3800]])

##################
## Division (/) ##
##################
'''Same as tensor.div(val)'''

print(tensor_vector / 4)
# tensor([1.2500, 5.0000, 3.5000, 0.2500, 1.0000])

print(tensor_matrix / 0.5)
# tensor([[ 1.1000,  0.5400],
#         [ 1.3000,  0.5000],
#         [-0.6800,  0.9200]])

########################
## Floor division (/) ##
########################
'''Same as tensor.div(val, rounding_mod="floor")'''

print(tensor_vector // 4)
# tensor([1, 5, 3, 0, 1])

print(tensor_matrix // 0.5)
# tensor([[ 1.,  0.],
#         [ 1.,  0.],
#         [-1.,  0.]])

##################
## Remainder (%) #
##################
'''Same as tensor.remainder(val)'''

print(tensor_vector.remainder(2))
# tensor([1, 0, 0, 1, 0])

print(tensor_matrix.remainder(0.3))
# tensor([[0.2500, 0.2700],
#         [0.0500, 0.2500],
#         [0.2600, 0.1600]])

######################
## Exponential (**) ##
######################
'''Same as tensor.pow(val)'''

print(tensor_vector ** 2)
# tensor([ 25, 400, 196,   1,  16])

print(tensor_matrix ** (-3))
# tensor([[  6.0105,  50.8053],
#         [  3.6413,  64.0000],
#         [-25.4427,  10.2737]])

###################################
## Reassignment to update values ##
###################################

tensor_updated = tensor_vector ** 3
print(tensor_updated)
# tensor([ 125, 8000, 2744,    1,   64])

tensor_matrix = tensor_matrix * (-2)
print(tensor_matrix)
# tensor([[-1.1000, -0.5400],
#         [-1.3000, -0.5000],
#         [ 0.6800, -0.9200]])
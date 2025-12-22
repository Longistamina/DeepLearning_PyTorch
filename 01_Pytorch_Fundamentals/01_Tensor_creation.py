'''
TENSOR EXPLAINED SIMPLY

1. MATHEMATICS (The "Multilinear Machine")
Wiki: "An algebraic object describing a multilinear relationship."
- Simple View: A tensor is a "machine" that takes in vectors and outputs a value.
- The Rule: It is "multilinear," meaning it processes each input independently and proportionally.
- Basis Independence: The tensor is a geometric reality that exists on its own. 
  It does not depend on the coordinate system (the "basis") you choose to describe it.

2. PHYSICS (The "Universal Truth")
Wiki: "A concise framework for solving mechanics and general relativity."
- Simple View: Tensors describe physical properties (like stress, gravity, or magnetism) that stay the same even if you rotate your perspective.
- Tensor Fields: In the real world, tensors often change from point to point (like the stress throughout a bridge). 
  This "map" of tensors is called a Tensor Field.
- Why it matters: It allows physicists to write one equation that works regardless of how an observer is moving or tilted.

3. DEEP LEARNING / PYTORCH (The "Component Array")
Wiki: "Components form an array... thought of as a high-dimensional matrix."
- Simple View: PyTorch treats tensor as a way of representing data
  When we write down the numbers of a mathematical tensor into a grid, that grid is what we call a "Tensor" in code.
  Works like list, vector, matrix, etc.
- Hierarchy:
    - Rank 0: Scalar (Magnitude only)
    - Rank 1: Vector (Magnitude and Direction)
    - Rank 2: Matrix (Components of a linear map)
    - Rank n: High-dimensional data (e.g., video frames or neural weights)

SUMMARY TABLE:
- Math: A multilinear map between vector spaces.
- Physics: A coordinate-independent physical property (often a Field).
- PyTorch: An optimized array of components for high-speed calculation.

#############################################

1. Create a tensor:
   + Scalar 0D
   + Vector 1D
   + Matrix 2D
   + Tensor nD
   + From Numpy array
   
2. Random:
   + torch.rand()
   + torch.randint()
   + torch.randn()
   + torch.manual_seed()
   + Show an image created by a random 3D tensor
   
3. Zeros, Ones, Full:
   + torch.zeros()
   + torch.ones()
   + torch.full()
   
4. Range and LinSpace
   + torch.arange()
   + torch.linspace()
   
5. Tensor-like
   + torch.rand_like()
   + torch.randint_like()
   + torch.randn_like()
   + torch.zeros_like()
   + torch.ones_like()
   + torch.full_like()
'''

import torch

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- 1. Create a tensor ---------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#

###############
## Scalar 0D ##
###############

tensor_scalar = torch.tensor(28495.2)

print(tensor_scalar)
# tensor(28495.1992)

print(tensor_scalar.item()) # Convert to a Python scalar
# 28495.19921875

print(tensor_scalar.ndim) # the number of dimension that tensor has
# 0

print(tensor_scalar.shape)
# torch.Size([])

print(type(tensor_scalar))
# <class 'torch.Tensor'>

###############
## Vector 1D ##
###############

tensor_vector = torch.tensor([85.2, 25.75])

print(tensor_vector)
# tensor([85.2000, 25.7500])

print(tensor_vector.ndim)
# 1

print(tensor_vector.shape)
# torch.Size([2])
# 2 elements

###############
## Matrix 2D ##
###############
'''The names of matrix and tensor are uppercase in ML and DL, like MATRIX_1'''

tensor_MATRIX = torch.tensor(
  [
    [25, 43, 57],
    [33, 22, 11],
    [45, 6, 2],
    [10, 5, 9]
  ]
)

print(tensor_MATRIX)
# tensor([[25, 43, 57],
#         [33, 22, 11],
#         [45,  6,  2],
#         [10,  5,  9]])

print(tensor_MATRIX.ndim)
# 2

print(tensor_MATRIX.shape)
# torch.Size([4, 3])
# 4 rows x 3 columns

print(tensor_MATRIX[0])
# tensor([25, 43, 57])
# The 1st row

print(tensor_MATRIX[:, 1])
# tensor([43, 22,  6,  5])
# The 2nd column

###############
## Tensor nD ##
###############
'''The names of matrix and tensor are uppercase in ML and DL, like TENSOR'''

TENSOR_nD = torch.tensor(
  [
    [
      [2, 5, 4],
      [3, 2, 7]
    ],
    [
      [5.9, 2.8, 2],
      [7.6, 3.2, 3]
    ]
  ]
)
'''NOTE: both 2D matrices inside must share the same shape'''

print(TENSOR_nD)
# tensor([[[2.0000, 5.0000, 4.0000],
#          [3.0000, 2.0000, 7.0000]],

#         [[5.9000, 2.8000, 2.0000],
#          [7.6000, 3.2000, 3.0000]]])

print(TENSOR_nD.ndim)
# 3

print(TENSOR_nD.shape)
# torch.Size([2, 2, 3])
# 2 mats x 2 rows x 2 columns

######################
## From Numpy array ##
######################

import numpy as np

np.random.seed(42)
np_arr = np.random.uniform(10, 20, (3, 2, 4)).round(2)
print(np_arr)
# [[[13.75 19.51 17.32 15.99]
#   [11.56 11.56 10.58 18.66]]

#  [[16.01 17.08 10.21 19.7 ]
#   [18.32 12.12 11.82 11.83]]

#  [[13.04 15.25 14.32 12.91]
#   [16.12 11.39 12.92 13.66]]]

TENSOR_arr = torch.tensor(np_arr)
print(TENSOR_arr)
# tensor([[[13.7500, 19.5100, 17.3200, 15.9900],
#          [11.5600, 11.5600, 10.5800, 18.6600]],

#         [[16.0100, 17.0800, 10.2100, 19.7000],
#          [18.3200, 12.1200, 11.8200, 11.8300]],

#         [[13.0400, 15.2500, 14.3200, 12.9100],
#          [16.1200, 11.3900, 12.9200, 13.6600]]], dtype=torch.float64)

print(TENSOR_arr.shape)
# torch.Size([3, 2, 4])


#------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 1. Random ---------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#

##################
## torch.rand() ##
##################
'''Returns a tensor filled with random numbers from a uniform distribution on the interval  [0,1)'''

tensor_rand = torch.rand(3, 4)
print(tensor_rand)
# tensor([[0.8407, 0.3572, 0.3737, 0.0696],
#         [0.2026, 0.8119, 0.9744, 0.7132],
#         [0.5552, 0.7161, 0.2946, 0.2977]])

print(torch.rand(2, 3, 2))
# tensor([[[0.9121, 0.5904],
#          [0.5699, 0.1031],
#          [0.7474, 0.6204]],

#         [[0.1316, 0.2758],
#          [0.6310, 0.6595],
#          [0.7299, 0.7429]]])

#####################
## torch.randint() ##
#####################
'''Returns a tensor filled with random integers generated uniformly between [low, high)'''

tensor_randint = torch.randint(2, 10, size=(5,)) # size must be a tuple
print(tensor_randint)
# tensor([4, 6, 7, 5, 3])

print(torch.randint(5, 15, (3, 2)))
# tensor([[ 8, 13],
#         [ 9,  8],
#         [13,  8]])

###################
## torch.randn() ##
###################
'''Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).'''

tensor_randn = torch.randn(4)
print(tensor_randn)
# tensor([-0.3348,  0.5918, -1.8890,  1.2116])

print(torch.randn((3, 2, 2)))
# tensor([[[-1.0697, -1.3891],
#          [-0.9897, -0.7155]],

#         [[-0.9873, -0.7341],
#          [ 0.4954, -0.6950]],

#         [[ 0.5342, -1.3749],
#          [-0.4603,  1.3058]]])

#########################
## torch.manual_seed() ##
#########################
'''Set a seed for reproducible random values'''

torch.manual_seed(32)
tensor_random = torch.randn((2, 3))

print(tensor_random)
# tensor([[ 0.8651,  0.0284,  0.5256],
#         [-0.3633, -0.4169, -1.2650]])
'''Always reproduces that same set of values'''

#################################################
## Show an image created by a random 3D tensor ##
#################################################
'''
The RGB value must range from 0-255
# Red:   0 - 255 (256 values)
# Green: 0 - 255 (256 values)
# Blue:  0 - 255 (256 values)
=> In total, we have 256**3 = 16777216 (more than 16.7 million colors)

A Full HD image has 1920x1080 resolutions
=> Create a 3D tensor with shapes (3, 1920, 1080)
=> 3 matrices of 1920x1080, each matrix represents Red, Green and Blue
'''

import matplotlib.pyplot as plt

torch.random.manual_seed(452)
IMAGE_FHD = torch.randint(0, 256, (3, 1920, 1080), device="cpu")

print(IMAGE_FHD.shape)
# torch.Size([3, 1920, 1080])
# 3 channles
# 1920-pixel hight
# 1080-pixel wide

plt.imshow(IMAGE_FHD.permute(2, 1, 0).numpy()) # permutate to (Width, Height, Channel) to match numpy
plt.axis("off") # Hide the x/y axes
plt.show()


#------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ 3. Zeros and Ones -----------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#

###################
## torch.zeros() ##
###################

tensor_zeros = torch.zeros(size=(5, 3))

print(tensor_zeros)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

##################
## torch.ones() ##
##################

tensor_ones = torch.ones(size=(3,7))

print(tensor_ones)
# tensor([[1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1.]])

##################
## torch.full() ##
##################

tensor_full = torch.full(size=(3, 3), fill_value=2.5)

print(tensor_full)
# tensor([[2.5000, 2.5000, 2.5000],
#         [2.5000, 2.5000, 2.5000],
#         [2.5000, 2.5000, 2.5000]])


#-----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ 4. Arange and LinSpace -----------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#

####################
## torch.arange() ##
####################
'''
Returns a 1-D tensor of size=(end-start)/step with values from the interval [start, end) taken with common difference step beginning from start.

Fix the space between elements
'''

print(torch.arange(5))
# tensor([0, 1, 2, 3, 4])

print(torch.arange(10, 20))
# tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

print(torch.arange(1, 10, step=2))
# tensor([1, 3, 5, 7, 9])

print(torch.arange(1, 2, step=0.25))
# tensor([1.0000, 1.2500, 1.5000, 1.7500])

######################
## torch.linspace() ##
######################
'''Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.

Fix the size (length, or steps)
'''

tensor_linspace = torch.linspace(start=2, end=10, steps=20)

print(tensor_linspace)
# tensor([ 2.0000,  2.4211,  2.8421,  3.2632,  3.6842,  4.1053,  4.5263,  4.9474,
#          5.3684,  5.7895,  6.2105,  6.6316,  7.0526,  7.4737,  7.8947,  8.3158,
#          8.7368,  9.1579,  9.5789, 10.0000])

print(tensor_linspace.shape)
# torch.Size([20])

'''
So, torch.linspace() will create a tensor ranging from 2 to 10.
All the values are evenly spaced so that the total number of values is 20 (steps, or length)
'''
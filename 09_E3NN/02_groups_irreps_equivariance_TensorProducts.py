'''
https://www.youtube.com/watch?v=9rS8gtey_Ic&t=119s

1. Group and Representations (irreps)
2. Equivariance (and equivariant polynomials)
3. Composition, Addition and Multiplication
'''

from e3nn import o3
import torch


#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ 1. Group and Representations ------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
'''
A group defines a set of operations and how these operations compose together.
=> in this case, Group = {rotations, parity, translation}

A representation defines how the group acts on some vector space.
=> example: scalar, vectors, pseudovectors, etc

Representation D(g, x) - transformation of x by g:
(can be written as D(g)x)
    + g ∊ G
    + x ∊ V (vector space)
    + Linear D(g, x + y) = D(g, x) + D(g, y)
'''

# Assume we have a R9 array: [a1, a2, a3, a4, a5, a6, a7, a8, a9]
# The first 3 [a1, a2, a3] represent 3 scalars => Rotation will not affect it
# The next 3 [a4, a5, a6] represent vector v1 => Rotation will transform it
# The next 3 [a7, a8, a9] represent vector v2 => Rotation will also transform it
# Since v1 and v2 are independent => Rotation will transform them independently
#
# The Rotation transformation is represented by this matrix D (9x9)
# [[1 0 0 0 0 0 0 0 0]   [a1
#  [0 1 0 0 0 0 0 0 0]    a2
#  [0 0 1 0 0 0 0 0 0]    a3
#  [0 0 0 1 2 3 0 0 0]    a4
#  [0 0 0 4 6 5 0 0 0] @  a5
#  [0 0 0 9 8 5 0 0 0]    a6
#  [0 0 0 0 0 0 1 2 2]    a7
#  [0 0 0 0 0 0 3 1 5]    a8
#  [0 0 0 0 0 0 3 4 1]]   a9]
#
# [[1 0 0 0 0 0 0 0 0]    [a1
#  [0 1 0 0 0 0 0 0 0] @   a2   => The 1 diagonal mean these scalars are transformed by Rotation
#  [0 0 1 0 0 0 0 0 0]     a3]
#
#  [0 0 0 1 2 3 0 0 0]    [a4
#  [0 0 0 4 6 5 0 0 0] @   a5    => The matrix is the rotation matrix, will transform the [a4, a5, a6] vector
#  [0 0 0 9 8 5 0 0 0]     a6]

irreps = o3.Irreps("3x0e + 2x1o") # Use Irreps to represent
                                       # 3x0e means 3 scalars
                                       # 2x1o means 2 vectors
                                       # (just representations, not hold any values yet)

# Initialize a rotation D group matrix from random alpha, beta, gamma
torch.manual_seed(42)
alpha, beta, gamma = torch.randn(3)

irreps.D_from_angles(alpha, beta, gamma)
# tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.8419,  0.0424,  0.5379,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0298,  0.9917, -0.1249,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  0.0000,  0.0000, -0.5388,  0.1212,  0.8337,  0.0000,  0.0000, 0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.8419,  0.0424, 0.5379],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0298,  0.9917, -0.1249],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.5388,  0.1212, 0.8337]])

'''
alpha, beta, gamma = Z-Y-Z Euler angle (https://www.youtube.com/watch?v=6DI77dgVOwM)

# alpha (α): A rotation around the original Z-axis.
# beta (β): A rotation around the new Y-axis (the axis that exists after the first rotation).
# gamma (γ): A rotation around the newest Z-axis (the axis that exists after the second rotation).

irreps.D_from_angles(alpha, beta, gamma)
=> computes Wigner D-matrix
=> describes how geometric objects with a specific angular momentum (such as scalars, vectors, or spherical harmonics)
   transform when the 3D space is rotated.
'''


#-----------------------------------------------------------------------------------------------------#
#------------------------------------------ 2. Equivariance ------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
'''
                      f
    R (1) -------------------------------> R' (3)
    |                                      |
    |                                      |
D(g)|                                 D'(g)|
    |                                      |
    |                                      |
    V                f                     V
    R (2) -------------------------------> R' (4)

(1) -> (2) -> (4) = f(D(g)x)
(1) -> (3) -> (4) = D'(g)f(x)

if f(D(g)x) = D'(g)f(x)
=> we achieve equivariance

########### EXPLAIN ###########

Assume we have an input x in R space (1), transformed by D(g)
-> output D(g)x is also in R (2)

The same, we have another input x' in R' (3), transformed by D'(g)
-> output D'(g)x' also in R' (4)

Assume we have another function f to transfrom x to x', (1) -> (3)
=> x' = f(x)

If we want to go from (1) to (4), there is two paths:
    + (1) -> (2) -> (4) = f(D(g)x)
    + (1) -> (3) -> (4) = D'(g)f(x)

if f(D(g)x) = D'(g)f(x)
=> we achieve equivariance

########################

Let's try to create an Equivariant Polynomial P(x) with e3nn

P(D(g)x) = D'(g)P(x)
'''

#################################################
## Equivariant Polynomial P(D(g)x) = D'(g)P(x) ##
#################################################
'''
Example of non-equivariant polynomial

[x       [x**2 + 2(y-z)
 y   ->   z**4 + 100xyz]
 z]

 This is not equivariant because when we rotate (transform) the [x, y, z],
 the output [x**2 + 2(y-z), z**4 + 100xyz] does not transform as a representation

 ###########################################

 Example of equivariant polynomial

 [x
  y   ->  [x**2 + y**2 + z**2]
  z]

[x**2 + y**2 + z**2] results in a scalar,
so when [x, y, z] is rotated, the scalar still stays the same
=> equivariant
(more specifically, this is invariant)
'''

#-----------------------------------------------------------------------------------------------------------------#
#---------------------------------- 3. Composition, Addition and Multiplication ----------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

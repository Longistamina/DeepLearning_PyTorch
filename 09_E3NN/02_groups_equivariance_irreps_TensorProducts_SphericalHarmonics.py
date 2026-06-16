'''
https://www.youtube.com/watch?v=9rS8gtey_Ic&t=119s

1. Group and Representations (irreps)
2. Equivariance (and equivariant polynomials)
3. Composition, Addition and Multiplication (reducible representations)
4. Irreducible representations: index, dimension, parity
'''

from e3nn import o3
import torch
import sympy as syp


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
                                       # 3x0e means 3 scalars (even parity)
                                       # 2x1o means 2 vectors (odd parity)
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

##################
## Compositions ##
##################
'''
f: V1 -> V2
h: V2 -> V3

if h and f are equivariant,
then the composition h(f(x)) is also equivariant

h(f(D1(g)x)) = h(D2(g)f(x)) = D3(g)h(f(x))
'''

##################
## Compositions ##
##################
'''
f: V1 -> V2
h: V2 -> V3

if h and f are equivariant,
then the addition h+f is also equivariant

D1(g)f(x) + D2(g)h(x) = D3(g)(f(x) + h(x))
'''

################################################################
## Multiplication (outer product) - Reducible representations ##
################################################################
'''
x: transform with Dx(g)
y: transform with Dy(g)

outer_product = x ⨂ y
(cartesian product)

This outer_product is equivariant

###################

However, the size and the dimension of the outer_product is much bigger than the original inputs
=> this growth in dimension can be problematic
=> Luckily, this outer_product is REDUCIBLE

A representation is reducible if you can decompose your vector space into smaller,
independent sub-spaces that each transform cleanly on their own

###################

How to reduce/decompose the outer_product?
=> into 3 components:
    + The trace (diagonal): the scalar product, stays the same after transformation
    + Anti-symmetric parts (the triu and tril): the cross-product of two vectors, a representation
                                                (transform when the two vectors transform)
    + Symmetric traceless, Degree of freedom: also a representation

###################

Theorem of multiplication decomposition

L1 ⨂ L2 = |L1 - L2| ⨁ ... ⨁ (L1 + L2)

Example:
    + 2 ⨂ 1 = 1 ⨁ 2 ⨁ 3
    + 2 ⨂ 2 = 0 ⨁ 1 ⨁ 2 ⨁ 3 ⨁ 4
'''

x1, x2, x3 = syp.symbols("x1 x2 x3")
y1, y2, y3 = syp.symbols("y1 y2 y3")

x = syp.Matrix([[x1, x2, x3]])
y = syp.Matrix([[y1, y2, y3]])

outer_product = x.T * y
print(outer_product)
# Matrix([[x1*y1, x1*y2, x1*y3],
#         [x2*y1, x2*y2, x2*y3],
#         [x3*y1, x3*y2, x3*y3]])
'''This outer_product is equivariant'''

'''
Let's reduce this outer_product
# Matrix([[x1*y1, x1*y2, x1*y3],
#         [x2*y1, x2*y2, x2*y3],
#         [x3*y1, x3*y2, x3*y3]])

The diagonal ~ trace = x1*y1 + x2*y2 + x3*y3
=> It's a scalar, we have
    + length L=1
    + stays the same after transformation

The anti-symmetric parts (the triu and tril) is the cross-product, a representation
[y1*z2 - z1*y2
 z1*x2 - x1*z2
 x1*y2 - y1*x2]
 => This is a vector
    + length L=3
    + transforms when the inputs transform (equivariant)

Symmetric traceless, Degree of freedom: also a representation
[c(x1*z2 + z1*x2)
 c(x1*y2 + y1*x2)
 2y1*y2 - x1*x2 - z1*z2
 c(y1*z2 + z1*y2)
 c(z1*z2 - x1*x2)
]
=> This is a vector
    + length L=5
    + transforms when the inputs transform (equivariant)

As we can see, the outer product 3x3 matrix has been decomposed into a scalar (L=1), and two vectors (L=3, L=5)
    3x3 = 1 + 3 + 5
'''


#------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 4. Irreducible representations -----------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
'''
An irreducible representation is the point at which you can no longer break the space down any further.
This is important because it isolates the simplest, most stable components of your data.

Characteristics of Irreducible representations

------
1. Indexing (l)

In the context of the rotation group SO(3), the index ``l`` (often called the angular momentum quantum number in physics)
serves as a label for the different ways a system can rotate

Why it matters: Instead of treating all data as a giant, undifferentiated blob of numbers,
                the index ``l`` groups numbers into specific "types."

The Hierarchy:
 l=0 is the simplest type (scalars),
 l=1 is slightly more complex (vectors),
 l=2 is even more complex (tensors), and so on.
 By using as an index, the neural network knows exactly how to apply rotation rules to each piece of data.

------
2. Dimensions (2l + 1)

This formula describes how many individual numbers (or "degrees of freedom")
are required to represent a feature of a specific type ``l``

For l=0 (scalar)
-> dimension = 2*0 + 1 = 1
-> you only need 1 number to represent a scalar type (that does not change under rotation)

For l=1 (vector)
-> dimension = 2*1 + 1 = 3
-> you need 3 numbers [x, y, z] to represent a vector in SO(3) space
-> the [x, y, z] will rotate together

For l=2 (higher space)
-> dimension = 2*2 + 1 = 5
-> need 5 numbers [x1, x2, x3, x4, x5] to represent that object in complex spatial shape

=>  This ensures that when your network processes data, it isn't just looking at random numbers;
    it’s looking at data with a fixed geometric structure!!!!

------
3. Parity
Parity is a symmetry operation that effectively flips the coordinates of your system (like looking at an object in a mirror)

Even (e) vs. Odd (o): When you add parity to the mix (making the group O(3)),
you aren't just labeling by ``l`` anymore;
you also care about what happens if you reflect the system.

Even representation: The values stay the same under reflection.
Odd representation: The values flip their sign (multiply by -1) under reflection.

Example:
    + A standard vector (like a velocity vector) is odd because if you mirror the world,
      the vector effectively points in the "opposite" direction.

    + A scalar (like temperature) is even because it doesn't care about the mirror-flip of the coordinate system.

-----------------

By keeping track of these labels (l), the size of the data (dimension), and the behavior under reflection (parity),
e3nn ensures that the neural network respects the fundamental physical laws of space and rotation.
'''

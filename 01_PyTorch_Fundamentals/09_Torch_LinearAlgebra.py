'''
1. Basic Operations: norm(), det(), inv(), matrix_power()
   + Magnitude and basic matrix properties.
   
2. Decompositions: cholesky(), qr(), svd(), eig(), eigh()
   + Breaking matrices into constituent parts.

3. Solvers & Inverses: solve(), lstsq(), pinv()
   + Solving linear systems of equations $Ax = b$.

4. Matrix Properties: matrix_rank(), cond()
   + Characterizing matrix behavior and stability.
'''

import torch

# Create sample square matrix (3x3)
A = torch.tensor([[4.0, 1.0, 1.0],
                  [1.0, 3.0, 1.0],
                  [1.0, 1.0, 2.0]])

# Create a vector for solvers
b = torch.tensor([1.0, 2.0, 3.0])

#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------ 1. Basic Operations --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.linalg.norm(A, ord=None): Computes matrix or vector norm.
torch.linalg.det(A): Computes the determinant of a square matrix.
torch.linalg.inv(A): Computes the inverse of a square matrix.
'''

# Frobenius norm (default for matrices)
print(torch.linalg.norm(A))
# tensor(5.83)

# Determinant
print(torch.linalg.det(A))
# tensor(17.00)

# Matrix Inverse
A_inv = torch.linalg.inv(A)
print(A_inv @ A) # Identity matrix


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------- 2. Decompositions ---------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.linalg.cholesky(A): Decomposes Hermitian, positive-definite matrix into $L L^H$.
torch.linalg.qr(A): QR decomposition.
torch.linalg.svd(A): Singular Value Decomposition.
torch.linalg.eig(A): Eigenvalues and eigenvectors.
'''

# Cholesky (requires positive-definite matrix)
L = torch.linalg.cholesky(A)
print(L)

# QR Decomposition
Q, R = torch.linalg.qr(A)

# SVD (Returns U, S, Vh)
U, S, Vh = torch.linalg.svd(A)
print(S) # Singular values


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Solvers & Inverses -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.linalg.solve(A, b): Computes the solution 'x' to $Ax = b$.
torch.linalg.lstsq(A, b): Computes the least-squares solution.
torch.linalg.pinv(A): Computes the Moore-Penrose pseudo-inverse.
'''

# Solve Ax = b
x = torch.linalg.solve(A, b)
print(x)
# tensor([-0.24,  0.41,  1.41])

# Pseudo-inverse (useful for non-square or singular matrices)
A_pinv = torch.linalg.pinv(A)


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------ 4. Matrix Properties -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
torch.linalg.matrix_rank(A): Computes the numerical rank.
torch.linalg.cond(A): Computes the condition number (stability check).
'''

print(torch.linalg.matrix_rank(A))
# tensor(3)

# Condition number (higher means more unstable/closer to singular)
print(torch.linalg.cond(A))
# tensor(3.58)
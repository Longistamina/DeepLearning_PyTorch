'''
1. Sum:
   + torch.sum(tensor)
   + tensor.sum()
   
2. Mean:
   + torch.mean(tensor)
   + tensor.mean()
   
3. Median:
   + torch.median(tensor)
   + tensor.median()
   
4. Mode:
   + torch.mode(tensor)
   + tensor.mode()
   
5. Variance:
   + torch.var(tensor)
   + tensor.var()
   
6. STD:
   + torch.std(tensor)
   + tensro.std()
   
7. Min:
   + torch.min(tensor)
   + tensor.min()
   
8. Max:
   + torch.max(tensor)
   + tensor.max()
   
9. Quantile:
   + torch.quantile(tensor)
   + tensor.quantile()
'''

import torch

tensor_v = torch.tensor([1, 3, 2, 4.5])

tensor_M = torch.tensor(
    [
        [2, 5, 4, 9],
        [3.2, 1, 5.4, 6.2]
    ]
)


#----------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Sum ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

print(torch.sum(tensor_v)) # tensor(10.5000)
print(tensor_v.sum())      # tensor(10.5000)

print(torch.sum(tensor_M)) # tensor(35.8000)
print(tensor_M.sum())      # tensor(35.8000)

print(torch.sum(tensor_M, dim=0)) # sum vertically
# tensor([ 5.2000,  6.0000,  9.4000, 15.2000])

print(tensor_M.sum(dim=1)) # sum horizontally
# tensor([20.0000, 15.8000])


#-----------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 2. Mean ------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

print(torch.mean(tensor_v)) # tensor(2.6250)
print(tensor_v.mean())      # tensor(2.6250)

print(torch.mean(tensor_M)) # tensor(4.4750)
print(tensor_M.mean())      # tensor(4.4750)

print(torch.mean(tensor_M, dim=0)) # mean vertically
# tensor([2.6000, 3.0000, 4.7000, 7.6000])

print(tensor_M.mean(dim=1)) # mean horizontally
# tensor([5.0000, 3.9500])


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 3. Median ------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

print(torch.median(tensor_v)) # tensor(2.)
print(tensor_v.median())      # tensor(2.)

print(torch.median(tensor_M)) # tensor(4.)
print(tensor_M.median())      # tensor(4.)

print(torch.median(tensor_M, dim=0)) # median vertically
# torch.return_types.median(
# values=tensor([2.0000, 1.0000, 4.0000, 6.2000]),
# indices=tensor([0, 1, 0, 1]))

print(tensor_M.median(dim=1)) # median horizontally
# torch.return_types.median(
# values=tensor([4.0000, 3.2000]),
# indices=tensor([2, 0]))


#-----------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 4. Mode ------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------#

print(torch.mode(tensor_v))
# torch.return_types.mode(
# values=tensor(1.),
# indices=tensor(0))

print(tensor_v.mode())
# torch.return_types.mode(
# values=tensor(1.),
# indices=tensor(0))

print(torch.mode(tensor_M)) # mode horizontally by default
# torch.return_types.mode(
# values=tensor([2., 1.]),
# indices=tensor([0, 1]))

print(tensor_M.mode()) # mode horizontally by default
# torch.return_types.mode(
# values=tensor([2., 1.]),
# indices=tensor([0, 1]))

print(torch.mode(tensor_M, dim=0)) # mode vertically
# torch.return_types.mode(
# values=tensor([2.0000, 1.0000, 4.0000, 6.2000]),
# indices=tensor([0, 1, 0, 1]))

print(tensor_M.mode(dim=1)) # mode horizontally
# torch.return_types.mode(
# values=tensor([2., 1.]),
# indices=tensor([0, 1]))


#---------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 5. Variance ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

print(torch.var(tensor_v)) # tensor(2.2292)
print(tensor_v.var())      # tensor(2.2292)

print(torch.var(tensor_M)) # tensor(6.3764)
print(tensor_M.var())      # tensor(6.3764)

print(torch.var(tensor_M, dim=0)) # var vertically
# tensor([0.7200, 8.0000, 0.9800, 3.9200])

print(tensor_M.var(dim=1)) # var horizontally
# tensor([8.6667, 5.4767])


#---------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 6. STD ---------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

print(torch.std(tensor_v)) # tensor(1.4930)
print(tensor_v.std())      # tensor(1.4930)

print(torch.std(tensor_M)) # tensor(2.5252)
print(tensor_M.std())      # tensor(2.5252)

print(torch.std(tensor_M, dim=0)) # std vertically
# tensor([0.8485, 2.8284, 0.9899, 1.9799])

print(tensor_M.std(dim=1)) # std horizontally
# tensor([2.9439, 2.3402])


#---------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 7. Min ---------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

print(torch.min(tensor_v)) # tensor(1.)
print(tensor_v.min())      # tensor(1.)

print(torch.min(tensor_M)) # tensor(1.)
print(tensor_M.min())      # tensor(1.)

print(torch.min(tensor_M, dim=0)) # min vertically
# torch.return_types.min(
# values=tensor([2.0000, 1.0000, 4.0000, 6.2000]),
# indices=tensor([0, 1, 0, 1]))

print(tensor_M.min(dim=1)) # min horizontally
# torch.return_types.min(
# values=tensor([2., 1.]),
# indices=tensor([0, 1]))

#---------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 8. Max ---------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

print(torch.max(tensor_v)) # tensor(4.5000)
print(tensor_v.max())      # tensor(4.5000)

print(torch.max(tensor_M)) # tensor(9.)
print(tensor_M.max())      # tensor(9.)

print(torch.max(tensor_M, dim=0)) # max vertically
# torch.return_types.max(
# values=tensor([3.2000, 5.0000, 5.4000, 9.0000]),
# indices=tensor([1, 0, 1, 0]))

print(tensor_M.max(dim=1)) # max horizontally
# torch.return_types.max(
# values=tensor([9.0000, 6.2000]),
# indices=tensor([3, 3]))


#--------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 9. Quantile ---------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------#

q_vals = torch.tensor([0.25, 0.75]) # q values must also be a torch tensor
                                    # Q1 and Q3

###########################

print(torch.quantile(tensor_v, q=q_vals)) # tensor([1.7500, 3.3750])
print(tensor_v.quantile(q=q_vals))        # tensor([1.7500, 3.3750])

###########################

torch.manual_seed(0)
tensor_Mb = torch.randint(low=1, high=21, size=(6, 10), dtype=torch.float32)
print(tensor_Mb)
# tensor([[ 5., 20., 14.,  1.,  4., 20.,  8.,  4., 18.,  4.],
#         [ 2.,  7., 17., 20., 19., 17., 17.,  9., 15., 14.],
#         [ 7., 20., 12., 15.,  5.,  2., 10., 10., 10.,  1.],
#         [ 2., 13.,  4.,  1., 16.,  6.,  3., 20., 12.,  9.],
#         [19.,  4., 17., 10., 12., 18.,  4., 16.,  3., 12.],
#         [ 1., 10., 14.,  2., 12.,  1.,  4.,  7., 17.,  8.]])

print(torch.quantile(tensor_Mb, q=q_vals)) # tensor([ 4., 16.])
print(tensor_Mb.quantile(q=q_vals))        # tensor([ 4., 16.])

torch.set_printoptions(linewidth=120)

print(torch.quantile(tensor_Mb, q=q_vals, dim=0)) # quantile vertically
# tensor([[ 2.0000,  7.7500, 12.5000,  1.2500,  6.7500,  3.0000,  4.0000,  7.5000, 10.5000,  5.0000],   Q1 array
#         [ 6.5000, 18.2500, 16.2500, 13.7500, 15.0000, 17.7500,  9.5000, 14.5000, 16.5000, 11.2500]])  Q3 array

print(torch.quantile(tensor_Mb, q=q_vals, dim=1)) # quantile horizontally
# tensor([[ 4.0000, 10.2500,  5.5000,  3.2500,  5.5000,  2.5000],    Q1 array
#         [17.0000, 17.0000, 11.5000, 12.7500, 16.7500, 11.5000]])   Q3 array
'''
1. Sort:
   + torch.sort(tensor)
   + tensor.sort()
   
2. ArgSort:
   + torch.argsort(tensor)
   + tensor.argsort()
   
3. ArgMin:
   + torch.argmin(tensor)
   + tensor.argmin()
   
4. ArgMax:
   + torch.argmax(tensor)
   + tensor.argmax()
'''

import torch

tensor_v = torch.tensor([1, 3, 2, 4.5, 0.7])

torch.manual_seed(0)
tensor_M = torch.randint(low=1, high=11, size=(4, 7))
print(tensor_M)
# tensor([[ 5, 10,  4,  1,  4, 10,  8],
#         [ 4,  8,  4,  2,  7,  7, 10],
#         [ 9,  7,  7,  9,  5,  4,  7],
#         [10,  2,  5,  5,  2, 10, 10]])


#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 1. Sort ---------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

####################
## Ascending sort ##
####################

print(torch.sort(tensor_v))
# torch.return_types.sort(
# values=tensor([0.7000, 1.0000, 2.0000, 3.0000, 4.5000]),
# indices=tensor([4, 0, 2, 1, 3]))

print(tensor_v.sort())
# torch.return_types.sort(
# values=tensor([0.7000, 1.0000, 2.0000, 3.0000, 4.5000]),
# indices=tensor([4, 0, 2, 1, 3]))

print(torch.sort(tensor_M, dim=1)) # Default sort horizontally
# values=tensor([[ 1,  4,  4,  5,  8, 10, 10],
#         [ 2,  4,  4,  7,  7,  8, 10],
#         [ 4,  5,  7,  7,  7,  9,  9],
#         [ 2,  2,  5,  5, 10, 10, 10]]),
# indices=tensor([[3, 2, 4, 0, 6, 1, 5],
#         [3, 0, 2, 4, 5, 1, 6],
#         [5, 4, 1, 2, 6, 0, 3],
#         [1, 4, 2, 3, 0, 5, 6]]))

print(tensor_M.sort(dim=0)) # sort vertically
# torch.return_types.sort(
# values=tensor([[ 4,  2,  4,  1,  2,  4,  7],
#         [ 5,  7,  4,  2,  4,  7,  8],
#         [ 9,  8,  5,  5,  5, 10, 10],
#         [10, 10,  7,  9,  7, 10, 10]]),
# indices=tensor([[1, 3, 0, 0, 3, 2, 2],
#         [0, 2, 1, 1, 0, 1, 0],
#         [2, 1, 3, 3, 2, 0, 1],
#         [3, 0, 2, 2, 1, 3, 3]]))

#####################
## Descending sort ##
#####################

print(torch.sort(tensor_v, descending=True))
# torch.return_types.sort(
# values=tensor([4.5000, 3.0000, 2.0000, 1.0000, 0.7000]),
# indices=tensor([3, 1, 2, 0, 4]))

print(tensor_v.sort(descending=True))
# torch.return_types.sort(
# values=tensor([4.5000, 3.0000, 2.0000, 1.0000, 0.7000]),
# indices=tensor([3, 1, 2, 0, 4]))

print(torch.sort(tensor_M, dim=1, descending=True)) # Default sort horizontally
# torch.return_types.sort(
# values=tensor([[10, 10,  8,  5,  4,  4,  1],
#         [10,  8,  7,  7,  4,  4,  2],
#         [ 9,  9,  7,  7,  7,  5,  4],
#         [10, 10, 10,  5,  5,  2,  2]]),
# indices=tensor([[1, 5, 6, 0, 2, 4, 3],
#         [6, 1, 4, 5, 0, 2, 3],
#         [0, 3, 1, 2, 6, 4, 5],
#         [0, 5, 6, 2, 3, 1, 4]]))

print(tensor_M.sort(dim=0, descending=True)) # sort vertically
# torch.return_types.sort(
# values=tensor([[10, 10,  7,  9,  7, 10, 10],
#         [ 9,  8,  5,  5,  5, 10, 10],
#         [ 5,  7,  4,  2,  4,  7,  8],
#         [ 4,  2,  4,  1,  2,  4,  7]]),
# indices=tensor([[3, 0, 2, 2, 1, 0, 1],
#         [2, 1, 3, 3, 2, 3, 3],
#         [0, 2, 0, 1, 0, 1, 0],
#         [1, 3, 1, 0, 3, 2, 2]]))


#-------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 2. ArgSort ---------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
'''Returns the indices that sort a tensor along a given dimension in ascending order by value.'''

#######################
## Ascending ArgSort ##
#######################

print(tensor_v)                # tensor([1.0000, 3.0000, 2.0000, 4.5000, 0.7000])
print(torch.argsort(tensor_v)) # tensor([4, 0, 2, 1, 3])
print(tensor_v.argsort())      # tensor([4, 0, 2, 1, 3])

################

print(tensor_M)
# tensor([[ 5, 10,  4,  1,  4, 10,  8],
#         [ 4,  8,  4,  2,  7,  7, 10],
#         [ 9,  7,  7,  9,  5,  4,  7],
#         [10,  2,  5,  5,  2, 10, 10]])

print(torch.argsort(tensor_M, dim=1)) # Default ArgSort horizontally
# tensor([[3, 2, 4, 0, 6, 1, 5],
#         [3, 0, 2, 4, 5, 1, 6],
#         [5, 4, 1, 2, 6, 0, 3],
#         [1, 4, 2, 3, 0, 5, 6]])

print(tensor_M.argsort(dim=0)) # ArgSort vertically
# tensor([[1, 3, 0, 0, 3, 2, 2],
#         [0, 2, 1, 1, 0, 1, 0],
#         [2, 1, 3, 3, 2, 0, 1],
#         [3, 0, 2, 2, 1, 3, 3]])

########################
## Descending ArgSort ##
########################

print(tensor_v)                                 # tensor([1.0000, 3.0000, 2.0000, 4.5000, 0.7000])
print(torch.argsort(tensor_v, descending=True)) # tensor([3, 1, 2, 0, 4])
print(tensor_v.argsort(descending=True))        # tensor([3, 1, 2, 0, 4])

################

print(tensor_M)
# tensor([[ 5, 10,  4,  1,  4, 10,  8],
#         [ 4,  8,  4,  2,  7,  7, 10],
#         [ 9,  7,  7,  9,  5,  4,  7],
#         [10,  2,  5,  5,  2, 10, 10]])

print(torch.argsort(tensor_M, dim=1, descending=True)) # Default ArgSort horizontally
# tensor([[1, 5, 6, 0, 2, 4, 3],
#         [6, 1, 4, 5, 0, 2, 3],
#         [0, 3, 1, 2, 6, 4, 5],
#         [0, 5, 6, 2, 3, 1, 4]])

print(tensor_M.argsort(dim=0, descending=True)) # ArgSort vertically
# tensor([[3, 0, 2, 2, 1, 0, 1],
#         [2, 1, 3, 3, 2, 3, 3],
#         [0, 2, 0, 1, 0, 1, 0],
#         [1, 3, 1, 0, 3, 2, 2]])


#------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 3. ArgMin ---------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
'''Returns the indices of the minimum value(s) of the flattened tensor or along a dimension'''

print(tensor_v) # tensor([1.0000, 3.0000, 2.0000, 4.5000, 0.7000])

print(torch.argmin(tensor_v)) # tensor(4)
print(tensor_v.argmin())      # tensor(4)

##########################################

print(tensor_M)
# tensor([[ 5, 10,  4,  1,  4, 10,  8],
#         [ 4,  8,  4,  2,  7,  7, 10],
#         [ 9,  7,  7,  9,  5,  4,  7],
#         [10,  2,  5,  5,  2, 10, 10]])

print(torch.argmin(tensor_M))    # tensor(3)
print(tensor_M.argmin(tensor_M)) # tensor(3)
'''It will flatten this 2D tensor into a 1D tensor, and perform argsort => 3 is the index of value 1 (min)'''

print(torch.argmin(tensor_M, dim=1)) # Find ArgMin horizontally
# tensor([3, 3, 5, 1])

print(tensor_M.argmin(dim=0)) # # Find ArgMin vertically
# tensor([1, 3, 0, 0, 3, 2, 2])


#------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- 4. ArgMax ---------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
'''Returns the indices of the maximum value(s) of the flattened tensor or along a dimension'''

print(tensor_v) # tensor([1.0000, 3.0000, 2.0000, 4.5000, 0.7000])

print(torch.argmax(tensor_v)) # tensor(3)
print(tensor_v.argmax())      # tensor(3)

##########################################

print(tensor_M)
# tensor([[ 5, 10,  4,  1,  4, 10,  8],
#         [ 4,  8,  4,  2,  7,  7, 10],
#         [ 9,  7,  7,  9,  5,  4,  7],
#         [10,  2,  5,  5,  2, 10, 10]])

print(torch.argmax(tensor_M))    # tensor(1)
print(tensor_M.argmax(tensor_M)) # tensor(1)
'''It will flatten this 2D tensor into a 1D tensor, and perform argsort => 1 is the index of value 10 (max)'''

print(torch.argmax(tensor_M, dim=1)) # Find ArgMax horizontally
# tensor([1, 6, 0, 0])

print(tensor_M.argmax(dim=0)) # # Find ArgMax vertically
# tensor([3, 0, 2, 2, 1, 0, 1])
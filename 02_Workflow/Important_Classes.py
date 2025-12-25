'''
torch.nn - contains all the building blocks for neural networks (computational graphs)
torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
torch.nn.Module - the base class for all neural network modules. (Should always subclass this class
                  (if you subclass nn.Module above, then should always overwrite forward())
'''
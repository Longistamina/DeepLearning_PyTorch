'''
1. Data preparation and loading
2. Build model
3. Fitting the model to data (training)
4. Evaluation
5. Making predictions (inference)
6. Saving and Loading a model
7. Put everything together

#############################################

torch.nn - contains all the building blocks for neural networks (computational graphs)

torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us

torch.nn.Module - the base class for all neural network modules. (Should always subclass this class
                  (if you subclass nn.Module above, then should always overwrite forward())
                
torch.optim - contains all the optimizers, helping us with gradient descent

torch.utils.data - contains tools for dataset handling
'''
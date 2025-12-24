'''
1. Device Status & Selection: is_available(), device_count(), current_device(), set_device()
   + Checking for GPU availability and managing active devices.

2. Device Properties: get_device_name(), get_device_capability(), get_device_properties()
   + Retrieving hardware specifications and metadata.

3. Memory Management: memory_allocated(), memory_reserved(), empty_cache()
   + Tracking and optimizing VRAM usage.

4. Synchronization & Streams: synchronize(), Stream(), Event()
   + Managing asynchronous execution and timing.

5. Random Number Generation (RNG): manual_seed(), manual_seed_all()
   + Ensuring reproducibility on GPU devices.
'''

import torch

#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 1. Device Status & Selection ---------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
Used to write "device-agnostic" code that works on both CPU and GPU.
'''

# Check if a CUDA-capable GPU is available
print(torch.cuda.is_available())
# Output: True (if GPU is present)

# Get the number of available GPUs
print(torch.cuda.device_count())

if torch.cuda.is_available():
    # Get index of the currently selected device (default is 0)
    print(torch.cuda.current_device()) 
    
    # Switch active GPU (useful for multi-GPU setups)
    # torch.cuda.set_device(1) 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# cuda


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------ 2. Device Properties -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

if torch.cuda.is_available():
    # Get the name of the GPU (e.g., 'NVIDIA GeForce RTX 3080')
    print(torch.cuda.get_device_name(0))
    
    # Get compute capability (Major, Minor)
    print(torch.cuda.get_device_capability(0)) 
    
    # Get detailed hardware properties (total memory, multi-processor count)
    print(torch.cuda.get_device_properties(0))


#-----------------------------------------------------------------------------------------------------------#
#------------------------------------------ 3. Memory Management -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
PyTorch uses a caching allocator to speed up memory allocations.
'''

if torch.cuda.is_available():
    # Create a tensor on GPU to use some memory
    x = torch.randn(1000, 1000).cuda()
    
    # Current memory occupied by tensors (in bytes)
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Total memory managed by the caching allocator (includes empty cache)
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Freeing all unoccupied cached memory (useful if you hit OOM errors)
    del x
    torch.cuda.empty_cache()


#-----------------------------------------------------------------------------------------------------------#
#-------------------------------------- 4. Synchronization & Streams ---------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
'''
CUDA operations are asynchronous. Use synchronize() to wait for completion.
'''

if torch.cuda.is_available():
    # Wait for all kernels on all streams on a specific device to finish
    torch.cuda.synchronize()

    # Timing CUDA operations using Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # ... GPU Operation ...
    end_event.record()

    torch.cuda.synchronize() # Must sync before calculating time
    print(f"Elapsed time: {start_event.elapsed_time(end_event)} ms")


#-----------------------------------------------------------------------------------------------------------#
#---------------------------------- 5. Random Number Generation (RNG) --------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

# Set seed for the current GPU
torch.cuda.manual_seed(42)

# Set seed for ALL available GPUs (important for distributed training)
torch.cuda.manual_seed_all(42)


#-----------------------------------------------------------------------------------------------------------#
#--------------------------------------- 6. Basic Data Movement --------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensor to device
tensor_cpu = torch.randn(3, 3)
tensor_gpu = tensor_cpu.to(device)

# Move entire model to device
# model = MyModel().to(device)
'''
ResNet-50 Architecture
+----------+------------------+--------------------------------+------------------+
| Layer    | Type             | Configuration                  | Output Size      |
+----------+------------------+--------------------------------+------------------+
| Input    | Image            | 224 x 224 x 3 (RGB)            | 224 x 224 x 3    |
|          |                  |                                |                  |
| Conv1    | Convolution      | 64 filters (7x7), Stride 2     | 112 x 112 x 64   |
| BN1      | Batch Norm       | -                              | 112 x 112 x 64   |
| ReLU     | Activation       | -                              | 112 x 112 x 64   |
| MaxPool  | Max Pooling      | 3x3 window, Stride 2           | 56 x 56 x 64     |
|          |                  |                                |                  |
| Stage 1  | Residual Block 1 | [1x1,64] [3x3,64] [1x1,256] ×3 | 56 x 56 x 256    |
|          |                  | + Skip Connection              |                  |
|          |                  |                                |                  |
| Stage 2  | Residual Block 2 | [1x1,128][3x3,128][1x1,512] ×4 | 28 x 28 x 512    |
|          |                  | + Skip Connection (stride 2)   |                  |
|          |                  |                                |                  |
| Stage 3  | Residual Block 3 | [1x1,256][3x3,256][1x1,1024]×6 | 14 x 14 x 1024   |
|          |                  | + Skip Connection (stride 2)   |                  |
|          |                  |                                |                  |
| Stage 4  | Residual Block 4 | [1x1,512][3x3,512][1x1,2048]×3 | 7 x 7 x 2048     |
|          |                  | + Skip Connection (stride 2)   |                  |
|          |                  |                                |                  |
| AvgPool  | Global Avg Pool  | 7x7 window                     | 1 x 1 x 2048     |
| Flatten  | Flatten          | -                              | 2048             |
| FC       | Fully Connected  | 1000 Neurons (Softmax)         | 1000             |
+----------+------------------+--------------------------------+------------------+

Residual Block (Bottleneck) Structure:
┌─────────────────────────────────────────────────┐
│  Input (x)                                      │
│    │                                            │
│    ├──────────────────────────┐                 │
│    │                          │                 │
│    ▼                          │ (Skip/Identity) │
│  1x1 Conv → BN → ReLU         │                 │
│    │                          │                 │
│    ▼                          │                 │
│  3x3 Conv → BN → ReLU         │                 │
│    │                          │                 │
│    ▼                          │                 │
│  1x1 Conv → BN                │                 │
│    │                          │                 │
│    └──────────► ADD ◄─────────┘                 │
│                 │                               │
│                 ▼                               │
│               ReLU                              │
│                 │                               │
│              Output                             │
└─────────────────────────────────────────────────┘

Key Characteristics of ResNet:

- Residual/Skip Connections: The core innovation - adds input directly to output
  F(x) + x instead of just F(x), solving the degradation problem
  
- Bottleneck Design: Uses 1x1 convolutions to reduce/restore dimensions, making
  the network more efficient (1x1 reduces → 3x3 processes → 1x1 expands)
  
- Batch Normalization: Applied after every convolutional layer before activation

- No Dropout: ResNet doesn't use dropout; residual connections provide 
  regularization effect
  
- Identity Mapping: When dimensions change, uses 1x1 convolutions to match 
  dimensions for the skip connection

- Depth Variants: ResNet-18, ResNet-34 (basic blocks), ResNet-50, ResNet-101, 
  ResNet-152 (bottleneck blocks)

- Total Parameters: ~25.6 million (ResNet-50)

The skip connections allow gradients to flow directly through the network during
backpropagation, enabling training of very deep networks (100+ layers) without
degradation. The network learns residual functions F(x) = H(x) - x rather than
directly learning H(x), which is easier to optimize.
'''
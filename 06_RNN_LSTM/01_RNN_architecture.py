'''
RNN (Recurrent Neural Network) Architecture
============================================

1. CORE CONCEPT
   - Processes sequential data by maintaining a "memory" of previous inputs
   - Same weights are applied at each time step (weight sharing)
   - Hidden state acts as memory, carrying information forward

2. BASIC STRUCTURE

   Input Sequence:  x₁ → x₂ → x₃ → ... → xₜ
                    ↓    ↓    ↓         ↓
   Hidden States:  h₁ → h₂ → h₃ → ... → hₜ
                    ↓    ↓    ↓         ↓
   Outputs:        y₁   y₂   y₃        yₜ

3. MATHEMATICAL FORMULATION

   At each time step t:
   
   hₜ = f(Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + bₕ)
   yₜ = g(Wₕᵧ · hₜ + bᵧ)
   
   Where:
   - hₜ = hidden state at time t
   - xₜ = input at time t
   - yₜ = output at time t
   - Wₕₕ = hidden-to-hidden weight matrix
   - Wₓₕ = input-to-hidden weight matrix
   - Wₕᵧ = hidden-to-output weight matrix
   - bₕ, bᵧ = bias vectors
   - f = activation function (usually tanh or ReLU)
   - g = output activation (softmax, sigmoid, or linear)

4. KEY COMPONENTS

   ┌─────────────────────────────────────┐
   │         RNN Cell at time t          │
   │                                     │
   │  hₜ₋₁ ──→ [●] ──→ hₜ                │
   │            ↑                        │
   │            │                        │
   │           xₜ                        │
   │            │                        │
   │            ↓                        │
   │           yₜ                        │
   └─────────────────────────────────────┘

5. STACKED RNN WITH 3 LAYERS (num_layers=3)

Input Sequence:    x₁ ──→ x₂ ──→ x₃ ──→ ... ──→ xₜ
                   ↓       ↓       ↓              ↓
                                                   
LAYER 1:          h₁₁ ──→ h₁₂ ──→ h₁₃ ──→ ... ──→ h₁ₜ
                   ↓       ↓       ↓              ↓
                                                   
LAYER 2:          h₂₁ ──→ h₂₂ ──→ h₂₃ ──→ ... ──→ h₂ₜ
                   ↓       ↓       ↓              ↓
                                                   
LAYER 3:          h₃₁ ──→ h₃₂ ──→ h₃₃ ──→ ... ──→ h₃ₜ
                   ↓       ↓       ↓              ↓
                                                   
Final Outputs:     y₁      y₂      y₃             yₜ

6. TRAINING

   - Backpropagation Through Time (BPTT)
   - Gradients computed backward through all time steps
   - Loss accumulated across sequence

8. CHALLENGES

   - Vanishing/exploding gradients in long sequences
   - Difficulty capturing long-range dependencies
   - Sequential processing (cannot parallelize easily)
   
   Solutions: LSTM/GRU, gradient clipping, better initialization

#####################################################################################

SIMPLE RNN EXAMPLE WITH NUMBERS
================================

TASK: Predict next number in sequence [1, 2, 3, ?]

SETUP
-----
- Input dimension: 1 (just the number itself)
- Hidden dimension: 2 (2 hidden units)
- Output dimension: 1 (predicted next number)
- Activation: tanh for hidden, linear for output
- Initial hidden state h₀ = [0, 0]

WEIGHTS (randomly initialized, then learned)
---------------------------------------------
Wₓₕ (input to hidden, 2×1):    [0.5]
                                [0.3]

Wₕₕ (hidden to hidden, 2×2):   [0.4  0.2]
                                [0.1  0.6]

Wₕᵧ (hidden to output, 1×2):   [0.7  0.5]

Biases:
bₕ = [0.1, 0.1]
bᵧ = 0.2


FORWARD PASS STEP-BY-STEP
==========================

TIME STEP 1: Input x₁ = 1
--------------------------
h₀ = [0, 0]

Hidden state calculation:
h₁ = tanh(Wₓₕ · x₁ + Wₕₕ · h₀ + bₕ)

Breaking it down:
Wₓₕ · x₁ = [0.5] · 1 = [0.5]
            [0.3]       [0.3]

Wₕₕ · h₀ = [0.4  0.2] · [0] = [0]
            [0.1  0.6]   [0]   [0]

Sum: [0.5] + [0] + [0.1] = [0.6]
     [0.3]   [0]   [0.1]   [0.4]

h₁ = tanh([0.6, 0.4]) = [0.537, 0.380]

Output:
y₁ = Wₕᵧ · h₁ + bᵧ
y₁ = [0.7  0.5] · [0.537] + 0.2
                   [0.380]
y₁ = 0.7×0.537 + 0.5×0.380 + 0.2
y₁ = 0.376 + 0.190 + 0.2 = 0.766


TIME STEP 2: Input x₂ = 2
--------------------------
h₁ = [0.537, 0.380]  (carried from previous step)

Hidden state calculation:
h₂ = tanh(Wₓₕ · x₂ + Wₕₕ · h₁ + bₕ)

Wₓₕ · x₂ = [0.5] · 2 = [1.0]
            [0.3]       [0.6]

Wₕₕ · h₁ = [0.4  0.2] · [0.537] = [0.291]
            [0.1  0.6]   [0.380]   [0.282]

Sum: [1.0] + [0.291] + [0.1] = [1.391]
     [0.6]   [0.282]   [0.1]   [0.982]

h₂ = tanh([1.391, 0.982]) = [0.884, 0.753]

Output:
y₂ = Wₕᵧ · h₂ + bᵧ
y₂ = [0.7  0.5] · [0.884] + 0.2
                   [0.753]
y₂ = 0.7×0.884 + 0.5×0.753 + 0.2
y₂ = 0.619 + 0.377 + 0.2 = 1.196


TIME STEP 3: Input x₃ = 3
--------------------------
h₂ = [0.884, 0.753]

Hidden state calculation:
h₃ = tanh(Wₓₕ · x₃ + Wₕₕ · h₂ + bₕ)

Wₓₕ · x₃ = [0.5] · 3 = [1.5]
            [0.3]       [0.9]

Wₕₕ · h₂ = [0.4  0.2] · [0.884] = [0.504]
            [0.1  0.6]   [0.753]   [0.540]

Sum: [1.5] + [0.504] + [0.1] = [2.104]
     [0.9]   [0.540]   [0.1]   [1.540]

h₃ = tanh([2.104, 1.540]) = [0.971, 0.914]

Output (PREDICTION for x₄):
y₃ = Wₕᵧ · h₃ + bᵧ
y₃ = [0.7  0.5] · [0.971] + 0.2
                   [0.914]
y₃ = 0.7×0.971 + 0.5×0.914 + 0.2
y₃ = 0.680 + 0.457 + 0.2 = 1.337


SUMMARY
=======
Input sequence:    [1,     2,     3    ]
Hidden states:     [h₁,    h₂,    h₃   ]
Outputs:           [0.766, 1.196, 1.337]
Target (learning): [2,     3,     4    ]

KEY OBSERVATIONS:
-----------------
1. Same weights (Wₓₕ, Wₕₕ, Wₕᵧ) used at every time step
2. Hidden state hₜ carries information from previous steps
3. Each hidden state depends on:
   - Current input xₜ
   - Previous hidden state hₜ₋₁
4. The network would learn better weights through backpropagation
   to make predictions closer to [2, 3, 4]
'''
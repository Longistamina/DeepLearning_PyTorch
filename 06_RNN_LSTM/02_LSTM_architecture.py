'''
LSTM (Long Short-Term Memory) Architecture
===========================================

1. CORE CONCEPT
   - Advanced RNN that solves vanishing gradient problem
   - Uses gating mechanisms to control information flow
   - Maintains both hidden state (h) and cell state (c)
   - Cell state acts as a "highway" for long-term memory

2. THE PROBLEM LSTM SOLVES
   - Regular RNNs struggle with long sequences (vanishing gradients)
   - LSTM can learn long-range dependencies (100+ time steps)
   - Gates allow selective memory: remember, forget, or update

3. BASIC STRUCTURE

   Input Sequence:  x₁ → x₂ → x₃ → ... → xₜ
                    ↓    ↓    ↓         ↓
   Cell States:    c₁ → c₂ → c₃ → ... → cₜ
   Hidden States:  h₁ → h₂ → h₃ → ... → hₜ
                    ↓    ↓    ↓         ↓
   Outputs:        y₁   y₂   y₃        yₜ

4. LSTM CELL ARCHITECTURE

   Legend: [×] = multiply  [+] = add  [σ] = sigmoid  [tanh] = tanh
   
   ╔═══════════════════════════════════════════════════════════════════╗
   ║                        LSTM MEMORY CELL                           ║
   ║                                                                   ║
   ║          Forget gate     Input gate               Output gate     ║
   ║   Cₜ₋₁ ─────[×]────────────[+]───────-───────────────┬────→ Cₜ----║------> Cₜ
   ║             ↑               ↑                        v            ║
   ║             │               │                      [tanh]         ║
   ║             │               │                        │            ║
 --║--- hₜ₋₁ ─→ [σ]      [σ]--->[×] <-[tanh]  [σ]-───--─>[×]─────→ hₜ--║------> hₜ
   ║      │      ↑        ↑      ↑     ↑        ↑         │        ↑   ║ |
   ║      │      │        │      │     │        │         |        │   ║ |
   ║      └──────┴────────┴──────┴─────┴────────┴─────────┘        │   ║ |
   ║             │        │      │     │        │                  │   ║ |
   ║   xₜ ───────┴────────┴──────┴─────┴────────┴──────────────────┘   ║ |
   ║                                                                   ║ |
   ╚═══════════════════════════════════════════════════════════════════╝ |
                                                                         V
                                                                         yₜ


   INFORMATION FLOW (Left to Right):
   ═══════════════════════════════════
   
   Step 1: FORGET GATE - What to keep from Cₜ₋₁
   ─────────────────────────────────────────────
           ┌────┐
   Cₜ₋₁ ──→│ ×  │ Element-wise multiply
           └─┬──┘
             ↑
          ┌──┴──┐
   hₜ₋₁ ─→│  σ  │ Forget gate: fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)
   xₜ ───→└─────┘ Outputs 0 (forget) to 1 (keep)
   
   
   Step 2: INPUT GATE - What new info to add
   ──────────────────────────────────────────
          ┌──────┐
          │tanh  │ Candidate: c̃ₜ = tanh(Wc·[hₜ₋₁,xₜ] + bc)
          └───┬──┘ New candidate values (-1 to 1)
              ↓
          ┌───┴──┐
          │  ×   │ Element-wise multiply
          └───┬──┘
              ↑
          ┌───┴──┐
   hₜ₋₁ ─→│  σ   │ Input gate: iₜ = σ(Wi·[hₜ₋₁,xₜ] + bi)
   xₜ ───→└──────┘ How much to add (0 to 1)
   
   
   Step 3: UPDATE CELL STATE
   ──────────────────────────
                 ┌───┐
   (fₜ × Cₜ₋₁) ─→│ + │──→ Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙c̃ₜ
   (iₜ × c̃ₜ)   ─→└───┘    (forget old + add new)
   
   
   Step 4: OUTPUT GATE - What to reveal
   ─────────────────────────────────────
          ┌──────┐
     Cₜ ─→│ tanh │ Squash cell state
          └───┬──┘
              ↓
          ┌───┴──┐
          │  ×   │──→ hₜ = oₜ ⊙ tanh(Cₜ)
          └───┬──┘    Filtered output
              ↑
          ┌───┴──┐
   hₜ₋₁ ─→│  σ   │ Output gate: oₜ = σ(Wo·[hₜ₋₁,xₜ] + bo)
   xₜ ───→└──────┘ How much to output (0 to 1)

5. FOUR GATES IN LSTM

   a) FORGET GATE (fₜ): Decides what to forget from cell state
      - Values between 0 (forget) and 1 (keep)
   
   b) INPUT GATE (iₜ): Decides what new information to add
      - Values between 0 (ignore) and 1 (add)
   
   c) CANDIDATE VALUES (c̃ₜ): New candidate information
      - Values between -1 and 1
   
   d) OUTPUT GATE (oₜ): Decides what to output
      - Values between 0 (block) and 1 (pass)

6. MATHEMATICAL FORMULATION

   At each time step t:
   
   Combined input: [hₜ₋₁, xₜ] (concatenation)
   
   fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     # Forget gate
   iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     # Input gate
   c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  # Candidate values
   oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     # Output gate
   
   cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ          # New cell state
   hₜ = oₜ ⊙ tanh(cₜ)                # New hidden state
   
   yₜ = Wy · hₜ + by                 # Output
   
   Where:
   - σ = sigmoid function (outputs 0 to 1)
   - tanh = hyperbolic tangent (outputs -1 to 1)
   - ⊙ = element-wise multiplication (Hadamard product)
   - [hₜ₋₁, xₜ] = concatenation of previous hidden and current input

7. INFORMATION FLOW

   Step 1: FORGET - Remove irrelevant information from cell state
           cₜ₋₁ × fₜ → partially forget old cell state
   
   Step 2: INPUT - Add new relevant information
           iₜ × c̃ₜ → what new info to add
   
   Step 3: UPDATE - Combine forget and input
           cₜ = (forget old) + (add new)
   
   Step 4: OUTPUT - Decide what to reveal
           hₜ = oₜ × tanh(cₜ)

8. STACKED LSTM WITH 2 LAYERS

Input Sequence:    x₁ ──→ x₂ ──→ x₃ ──→ ... ──→ xₜ
                   ↓       ↓       ↓              ↓
                                                   
LAYER 1:          c₁₁ ──→ c₁₂ ──→ c₁₃ ──→ ... ──→ c₁ₜ
                  h₁₁ ──→ h₁₂ ──→ h₁₃ ──→ ... ──→ h₁ₜ
                   ↓       ↓       ↓              ↓
                                                   
LAYER 2:          c₂₁ ──→ c₂₂ ──→ c₂₃ ──→ ... ──→ c₂ₜ
                  h₂₁ ──→ h₂₂ ──→ h₂₃ ──→ ... ──→ h₂ₜ
                   ↓       ↓       ↓              ↓
                                                   
Final Outputs:     y₁      y₂      y₃             yₜ

9. ADVANTAGES OVER VANILLA RNN
   - Solves vanishing gradient problem via cell state highway
   - Can learn dependencies 100+ steps back
   - Gates provide interpretability (what's being remembered/forgotten)
   - More stable training

10. CHALLENGES
    - More parameters than vanilla RNN (4× more weights)
    - Slower training and inference
    - Still sequential (hard to parallelize)
    
    Modern alternatives: GRU (simpler), Transformers (parallelizable)

#####################################################################################

LSTM EXAMPLE WITH NUMBERS
==========================

TASK: Predict next number in sequence [1, 2, 3, ?]

SETUP
-----
- Input dimension: 1 (just the number)
- Hidden dimension: 2 (2 hidden units)
- Cell state dimension: 2 (same as hidden)
- Output dimension: 1
- Initial states: h₀ = [0, 0], c₀ = [0, 0]

WEIGHTS (simplified for demonstration)
---------------------------------------
Note: In reality, each gate has separate weights for h and x.
For clarity, we'll show combined weight matrices after concatenation [hₜ₋₁, xₜ].
Concatenated input dimension: 2 + 1 = 3

Forget Gate Weights Wf (2×3):
[0.4  0.2  0.5]
[0.3  0.4  0.3]
bf = [0.1, 0.1]

Input Gate Weights Wi (2×3):
[0.5  0.3  0.4]
[0.2  0.5  0.3]
bi = [0.1, 0.1]

Candidate Weights Wc (2×3):
[0.6  0.2  0.5]
[0.3  0.5  0.4]
bc = [0.0, 0.0]

Output Gate Weights Wo (2×3):
[0.5  0.4  0.3]
[0.4  0.3  0.5]
bo = [0.1, 0.1]

Hidden-to-Output Weights Wy (1×2):
[0.7  0.5]
by = 0.2


FORWARD PASS STEP-BY-STEP
==========================

TIME STEP 1: Input x₁ = 1
--------------------------
Initial states: h₀ = [0, 0], c₀ = [0, 0]
Concatenated input: [h₀, x₁] = [0, 0, 1]

FORGET GATE:
fₜ = σ(Wf · [0, 0, 1] + bf)

Wf · [0, 0, 1] = [0.4  0.2  0.5] · [0]   [0.5]
                  [0.3  0.4  0.3]   [0] = [0.3]
                                    [1]

fₜ = σ([0.5 + 0.1, 0.3 + 0.1]) = σ([0.6, 0.4])
fₜ = [0.646, 0.599]  ← sigmoid values

INPUT GATE:
iₜ = σ(Wi · [0, 0, 1] + bi)

Wi · [0, 0, 1] = [0.5  0.3  0.4] · [0]   [0.4]
                  [0.2  0.5  0.3]   [0] = [0.3]
                                    [1]

iₜ = σ([0.4 + 0.1, 0.3 + 0.1]) = σ([0.5, 0.4])
iₜ = [0.622, 0.599]

CANDIDATE VALUES:
c̃ₜ = tanh(Wc · [0, 0, 1] + bc)

Wc · [0, 0, 1] = [0.6  0.2  0.5] · [0]   [0.5]
                  [0.3  0.5  0.4]   [0] = [0.4]
                                    [1]

c̃ₜ = tanh([0.5, 0.4]) = [0.462, 0.380]

OUTPUT GATE:
oₜ = σ(Wo · [0, 0, 1] + bo)

Wo · [0, 0, 1] = [0.5  0.4  0.3] · [0]   [0.3]
                  [0.4  0.3  0.5]   [0] = [0.5]
                                    [1]

oₜ = σ([0.3 + 0.1, 0.5 + 0.1]) = σ([0.4, 0.6])
oₜ = [0.599, 0.646]

NEW CELL STATE:
c₁ = fₜ ⊙ c₀ + iₜ ⊙ c̃ₜ
c₁ = [0.646, 0.599] ⊙ [0, 0] + [0.622, 0.599] ⊙ [0.462, 0.380]
c₁ = [0, 0] + [0.287, 0.228]
c₁ = [0.287, 0.228]

NEW HIDDEN STATE:
h₁ = oₜ ⊙ tanh(c₁)
h₁ = [0.599, 0.646] ⊙ tanh([0.287, 0.228])
h₁ = [0.599, 0.646] ⊙ [0.279, 0.224]
h₁ = [0.167, 0.145]

OUTPUT:
y₁ = Wy · h₁ + by
y₁ = [0.7  0.5] · [0.167] + 0.2
                   [0.145]
y₁ = 0.7×0.167 + 0.5×0.145 + 0.2
y₁ = 0.117 + 0.073 + 0.2 = 0.390


TIME STEP 2: Input x₂ = 2
--------------------------
Previous states: h₁ = [0.167, 0.145], c₁ = [0.287, 0.228]
Concatenated input: [h₁, x₂] = [0.167, 0.145, 2]

FORGET GATE:
fₜ = σ(Wf · [0.167, 0.145, 2] + bf)

Wf · [0.167, 0.145, 2] = [0.4  0.2  0.5] · [0.167]   [1.096]
                          [0.3  0.4  0.3]   [0.145] = [0.708]
                                            [2]

fₜ = σ([1.096 + 0.1, 0.708 + 0.1]) = σ([1.196, 0.808])
fₜ = [0.768, 0.692]

INPUT GATE:
iₜ = σ(Wi · [0.167, 0.145, 2] + bi)

Wi · [0.167, 0.145, 2] = [0.5  0.3  0.4] · [0.167]   [0.970]
                          [0.2  0.5  0.3]   [0.145] = [0.706]
                                            [2]

iₜ = σ([0.970 + 0.1, 0.706 + 0.1]) = σ([1.070, 0.806])
iₜ = [0.745, 0.691]

CANDIDATE VALUES:
c̃ₜ = tanh(Wc · [0.167, 0.145, 2] + bc)

Wc · [0.167, 0.145, 2] = [0.6  0.2  0.5] · [0.167]   [1.129]
                          [0.3  0.5  0.4]   [0.145] = [0.923]
                                            [2]

c̃ₜ = tanh([1.129, 0.923]) = [0.810, 0.726]

OUTPUT GATE:
oₜ = σ(Wo · [0.167, 0.145, 2] + bo)

Wo · [0.167, 0.145, 2] = [0.5  0.4  0.3] · [0.167]   [0.741]
                          [0.4  0.3  0.5]   [0.145] = [1.110]
                                            [2]

oₜ = σ([0.741 + 0.1, 1.110 + 0.1]) = σ([0.841, 1.210])
oₜ = [0.699, 0.770]

NEW CELL STATE:
c₂ = fₜ ⊙ c₁ + iₜ ⊙ c̃ₜ
c₂ = [0.768, 0.692] ⊙ [0.287, 0.228] + [0.745, 0.691] ⊙ [0.810, 0.726]
c₂ = [0.220, 0.158] + [0.603, 0.502]
c₂ = [0.823, 0.660]

NEW HIDDEN STATE:
h₂ = oₜ ⊙ tanh(c₂)
h₂ = [0.699, 0.770] ⊙ tanh([0.823, 0.660])
h₂ = [0.699, 0.770] ⊙ [0.674, 0.578]
h₂ = [0.471, 0.445]

OUTPUT:
y₂ = Wy · h₂ + by
y₂ = [0.7  0.5] · [0.471] + 0.2
                   [0.445]
y₂ = 0.7×0.471 + 0.5×0.445 + 0.2
y₂ = 0.330 + 0.223 + 0.2 = 0.753


TIME STEP 3: Input x₃ = 3
--------------------------
Previous states: h₂ = [0.471, 0.445], c₂ = [0.823, 0.660]
Concatenated input: [h₂, x₃] = [0.471, 0.445, 3]

FORGET GATE:
fₜ = σ(Wf · [0.471, 0.445, 3] + bf)

Wf · [0.471, 0.445, 3] = [0.4  0.2  0.5] · [0.471]   [1.867]
                          [0.3  0.4  0.3]   [0.445] = [1.219]
                                            [3]

fₜ = σ([1.867 + 0.1, 1.219 + 0.1]) = σ([1.967, 1.319])
fₜ = [0.877, 0.789]

INPUT GATE:
iₜ = σ(Wi · [0.471, 0.445, 3] + bi)

Wi · [0.471, 0.445, 3] = [0.5  0.3  0.4] · [0.471]   [1.569]
                          [0.2  0.5  0.3]   [0.445] = [1.217]
                                            [3]

iₜ = σ([1.569 + 0.1, 1.217 + 0.1]) = σ([1.669, 1.317])
iₜ = [0.841, 0.789]

CANDIDATE VALUES:
c̃ₜ = tanh(Wc · [0.471, 0.445, 3] + bc)

Wc · [0.471, 0.445, 3] = [0.6  0.2  0.5] · [0.471]   [1.872]
                          [0.3  0.5  0.4]   [0.445] = [1.586]
                                            [3]

c̃ₜ = tanh([1.872, 1.586]) = [0.954, 0.922]

OUTPUT GATE:
oₜ = σ(Wo · [0.471, 0.445, 3] + bo)

Wo · [0.471, 0.445, 3] = [0.5  0.4  0.3] · [0.471]   [1.313]
                          [0.4  0.3  0.5]   [0.445] = [1.857]
                                            [3]

oₜ = σ([1.313 + 0.1, 1.857 + 0.1]) = σ([1.413, 1.957])
oₜ = [0.804, 0.876]

NEW CELL STATE:
c₃ = fₜ ⊙ c₂ + iₜ ⊙ c̃ₜ
c₃ = [0.877, 0.789] ⊙ [0.823, 0.660] + [0.841, 0.789] ⊙ [0.954, 0.922]
c₃ = [0.722, 0.521] + [0.802, 0.727]
c₃ = [1.524, 1.248]

NEW HIDDEN STATE:
h₃ = oₜ ⊙ tanh(c₃)
h₃ = [0.804, 0.876] ⊙ tanh([1.524, 1.248])
h₃ = [0.804, 0.876] ⊙ [0.909, 0.848]
h₃ = [0.731, 0.743]

OUTPUT (PREDICTION for x₄):
y₃ = Wy · h₃ + by
y₃ = [0.7  0.5] · [0.731] + 0.2
                   [0.743]
y₃ = 0.7×0.731 + 0.5×0.743 + 0.2
y₃ = 0.512 + 0.372 + 0.2 = 1.084


SUMMARY
=======
Input sequence:     [1,     2,     3    ]
Cell states:        [c₁,    c₂,    c₃   ]
Hidden states:      [h₁,    h₂,    h₃   ]
Outputs:            [0.390, 0.753, 1.084]
Target (learning):  [2,     3,     4    ]

KEY OBSERVATIONS:
-----------------
1. Cell state (c) acts as long-term memory, flowing through time
2. Hidden state (h) is filtered version of cell state
3. Gates control information flow:
   - Forget gate: What to keep from previous cell state
   - Input gate: What new information to add
   - Output gate: What to reveal from cell state
4. Each gate uses sigmoid (0-1) for "how much"
5. Candidate uses tanh (-1 to 1) for "what value"
6. Cell state can grow unbounded, protected by gates
7. Through training, network learns:
   - When to forget (forget gate)
   - What to remember (input gate)
   - What to output (output gate)

GATE INTERPRETATION FOR THIS EXAMPLE:
--------------------------------------
Time Step 1: Learning starts
- Forget gate ~0.6: Moderately forgetting (but c₀=0 anyway)
- Input gate ~0.6: Moderately accepting new info
- Cell state starts building up memory

Time Step 2: Pattern recognition
- Forget gate ~0.7: Keeping most of previous memory
- Input gate ~0.7: Adding more new information
- Cell state growing (0.287→0.823)

Time Step 3: Confident prediction
- Forget gate ~0.8: Strongly keeping memory
- Input gate ~0.8: Strongly adding pattern info
- Cell state peaks (1.524), capturing sequence pattern
- Output closest to target so far

The cell state acts as a "conveyor belt" of information,
with gates deciding what gets on, what stays, and what gets off!
'''
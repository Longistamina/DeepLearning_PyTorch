'''
Title: Sequence-to-Sequence (Seq2Seq) Architecture Explained
Based on: StatQuest with Josh Starmer (Video: L8HKweZIOmg)

1. OVERVIEW
   The Seq2Seq model is a neural network architecture designed to convert one sequence of data (like words in English) 
   into another sequence of data (like words in French). It consists of two main parts: the Encoder and the Decoder.

   Example used in concept: Translating "So long" to "Au revoir".

2. THE ENCODER (The Reader)
   The Encoder's job is to read the input sequence and summarize it into a compact numerical representation.

   - Step A: Word Embeddings
     The raw words (e.g., "So", "long") are first converted into numbers. 
     Specifically, they are turned into "Embeddings"—coordinates in a low-dimensional space where similar words are close together.

   - Step B: The Unrolled RNN/LSTM
     The Encoder consists of a Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) units. 
     It processes the input one word at a time.
     
     1. The first unit takes the embedding for "So" and an initial hidden state (usually zeros).
     2. It performs some math to produce a new hidden state.
     3. This hidden state is passed to the next unit.
     4. The next unit takes the embedding for "long" and the hidden state from the previous step.
     5. It produces a final hidden state.

   - Step C: The Context Vector (The Handoff)
     The final hidden state (and cell state, if using LSTMs) produced after the last input word is processed 
     represents the summary of the entire input sentence. 
     This is often called the "Context Vector." It contains the "meaning" of the input sequence.

3. THE DECODER (The Writer)
   The Decoder's job is to take that summary and generate the output sequence one word at a time.

   - Step A: Initialization
     The Decoder is also an RNN/LSTM. Crucially, its initial hidden state is NOT set to zeros. 
     Instead, it is initialized with the Context Vector (the final state) from the Encoder. This connects the two networks.

   - Step B: The Start Token
     To kickstart the generation, the Decoder is fed a special token, often denoted as <EOS> (End of Sentence) or <SOS> (Start of Sentence) 
     depending on the specific convention, but typically a "Start" signal.

   - Step C: Generating Word 1
     1. The Decoder takes the Start token and the Context Vector.
     2. It calculates a new hidden state.
     3. This state is passed through a "Linear Layer" (to resize the data) and then a "Softmax" function.
     4. The Softmax produces a list of probabilities for every word in the vocabulary (e.g., 0.85 for "Au", 0.10 for "Bonjour", etc.).
     5. The word with the highest probability is selected as the output: "Au".

   - Step D: Generating Word 2
     1. The chosen word from the previous step ("Au") becomes the INPUT for the next step.
     2. The Decoder takes "Au" and the hidden state from the previous step.
     3. It repeats the math (Linear + Softmax).
     4. The highest probability word is selected: "revoir".

   - Step E: Termination
     The process repeats until the Decoder selects the special <EOS> (End of Sentence) token. This signals that the translation is complete.

4. SUMMARY OF FLOW
   Input Sequence -> [Encoder] -> Context Vector -> [Decoder] -> Output Sequence
'''
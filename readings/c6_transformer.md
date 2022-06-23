# Transformer

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Self-Attention](#11-self-attention)
  - [1.2. Position Encoding](#12-position-encoding)
  - [1.3. Transformer](#13-transformer)

# 1. Introduction
## 1.1. Self-Attention
- The attention mechanism is a “word to word” operation. It is actually a token-to-token operation, but we will keep it to the word level to keep the explanation simple. 
- The attention mechanism will find how each word is related to all other words in a sequence, including the word being analyzed itself. 
- Let’s examine the following sequence: `The cat sat on the mat.`
- Attention will run dot products between word vectors and determine the strongest relationships of a word with all the other words, including itself (“cat” and “cat”):
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/175253453-d99c43a1-799c-4595-915e-d9e192010301.png" width="250" />
</p>


## 1.2. Position Encoding
## 1.3. Transformer
- In December 2017, Google Brain and Google Research published the seminal Vaswani et al., Attention is All You Need paper. The Transformer was born. The Transformer outperformed the existing state-of-the-art NLP models.
- The idea of the attention head of the Transformer is to do away with recurrent neural network features.
### 1.3.1. Structure of Transformer
- The original Transformer model is a stack of 6 layers (6-layer encoder stack on the left and a 6-layer decoder stack on the right). 
  - The output of layer `l` is the input of layer `l+1` until the final prediction is reached. 
- On the left, the inputs enter the encoder side of the Transformer through an attention sublayer and a feedforward sublayer.
- On the right, the target outputs go into the decoder side of the Transformer through two attention sublayers and a feedforward network sublayer.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/174724427-238a19c9-77d8-443c-8366-e3cf7070fd83.png" width="300" />
</p>

### 1.3.2. Encoder Stack
- The layers of the encoder and decoder of the original Transformer model are stacks of layers. 
- Each layer contains **two main sublayers**: 
  - A `multi-headed attention` mechanism 
  - A `fully connected position-wise` feedforward network.
- **Residual connection** surrounds each main `sublayer`
  - These connections transport the unprocessed input x of a sublayer to a layer normalization function. 
  - This way, we are certain that key information such as positional encoding is not lost on the way. The normalized output of each layer is thus:
  - `LayerNormalization (x + Sublayer(x))`
- **Multi-head attention** mechanisms perform the same functions from layer 1 to 6. 
  - However, they do not perform the same tasks. Each layer learns from the previous layer and explores different ways of associating the tokens in the sequence.
- `d_model` the output of every sublayer of the model has a constant dimension, including the embedding layer and the residual connections. 
  - Note: Original Transformer architecture, d_model = 512.
  - d_model has a powerful consequence. Practically all the key operations are dot products. As a result, the dimensions remain stable, which reduces the number of operations to calculate, reduces machine consumption, and makes it easier to trace the information as it flows through the model.




<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/175254111-1b48577e-948c-4e2b-a417-baab6c64d65f.png" width="200" />
</p>

[(Back to top)](#table-of-contents)

# Self-Attention
- RNN: to compute 2nd time stamp, we need to wait for the output of the 1st time stamp to finish computation.
  - Not efficinet for a long sequence
- CNN: only cares about window-based model, and then integrate the local feature from each layers
  - Only can consider the local features, not global features 
  - <img width="1161" alt="Screenshot 2022-05-18 at 19 26 17" src="https://user-images.githubusercontent.com/64508435/169028216-2433aee0-6f0b-4174-96b7-14c1e88e47b4.png">
- Attention: has to do with 2 sentence. How about if we only have 1 sentence ? => Self-Attention
- Self-attention: (similar to attention in Seq2Seq model but v)
- For each word, we will has 2 vectors, `k` and `v`.
  - `k`: how to find the weight of the word (to decide the important of the word in the context)
  - `v`: contain the meaning of the word
# Position Encoding
- Normalization ? Change the index to between 0 and 1.
  - Issue: relative distance will change -> varying time-stamp

How to solve ? There are 3 principles
- Unique encoding for each location
- Relative position distance should be invariant in difference sentence length 
  - Relative distance positive should be same (no matter of indexing)  
- consistent upper/lower bound (between 0 and 1)
- **Solution**:  Periodic function! e.g., Sinusoidal position encoding

# Transformer:
- Based on Encoder-Decoder 

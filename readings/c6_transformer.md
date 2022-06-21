# Transformer

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Self-Attention](#11-self-attention)
  - [1.2. Position Encoding](#12-position-encoding)
  - [1.3. Transformer](#13-transformer)

# 1. Introduction
## 1.1. Self-Attention
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

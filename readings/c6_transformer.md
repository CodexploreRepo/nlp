# Transformer

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.1. Self-Attention](#11-self-attention)
  - [1.3. Transformer](#13-transformer)
- [2. Encoder Stack of Transformer](#2-encoder-stack-of-transformer)
  - [2.1. Input embedding](#21-input-embedding) 
  - [2.2. Position Encoding](#22-position-encoding)
  - [2.3. Adding Positional Encoding to Input Embedding Vector](#23-adding-positional-encoding-to-input-embedding-vector)


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

# 2. Encoder Stack of Transformer
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

## 2.1. Input embedding
- The **input embedding** sublayer converts the *input tokens* to vectors of dimension *dmodel = 512* using learned embeddings in the original Transformer model.
- A **tokenizer** will transform a sentence into tokens. 
  - Each tokenizer has its methods, such as BPE, word piece, and sentence piece methods. The Transformer initially used BPE, but other models use other methods.
  - `the Transformer is an innovative NLP model!` &#8594; `['the', 'transform', 'er', 'is', 'an', 'innovative', 'n', 'l', 'p', 'model', '!']`
  - Note 1: tokenizer normalized the string to *lowercase* and *truncated* it into subparts. 
  - Note 2: tokenizer will generally provide an *integer representation* that will be used for the embedding process. 
  ```Python
  text = "The cat slept on the couch.It was too tired to get up."
  tokenized text= [1996, 4937, 7771, 2006, 1996, 6411, 1012, 2009, 2001, 2205, 5458, 2000, 2131, 2039, 1012]
  ```
- The Transformer contains a learned embedding sublayer. Many embedding methods can be applied to the tokenized input.

## 2.2. Position Encoding
- The idea is to add a positional encoding value to the input embedding instead of having additional vectors to describe the position of a token in a sequence.
- There are many ways to achieve positional encoding. This section will focus on the designers’ clever way to use a unit sphere to represent positional encoding with sine and cosine values that will thus remain small but useful.

<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/175315787-bd88472b-9c1c-4d97-9c77-5b22a1366cf4.png" width="300" /><br>
<img src="https://user-images.githubusercontent.com/64508435/175316112-50c76601-66a0-4f2f-a10d-ce1b6d03724a.png" width="300" />
</p>

- From the above formula, the sine function will be applied to the even dim  and the cosine function to the odd dim.

```Python
def positional_encoding(pos,pe):
  for i in range(0, 512,2):
           pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
           pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
  return pe
```

## 2.3. Adding Positional Encoding to Input Embedding Vector
- The authors of the Transformer found a simple way by merely adding the positional encoding vector to the word embedding vector
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/175322405-7d851365-0d28-4f6c-9e01-f8a71ab3bad9.png" width="300" />
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

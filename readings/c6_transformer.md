
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

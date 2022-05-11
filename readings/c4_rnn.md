# Recurrent Neural Networks


# Table of contents
- [Table of contents](#table-of-contents)


# Test
-  Bag-of-words(BoW) models: Naive Bayes, Word2Vec Embedding  
  - Problem: we are ignore the order of words, tokens when counting
  
<p align="center"><img width="600" alt="Screenshot 2022-04-20 at 19 25 50" src="https://user-images.githubusercontent.com/64508435/166672704-500ad9d9-ff40-4379-b994-465c51859939.png"></p>

- Language Model: assigns prob to 
  - Either a sequence of words
  - or upcomming word given the preceeding words 
  - Why LM ? as we dont want to label any words
- Markov Assumption: given the current state, the future state is independent with past states. 
  - Given a couple of words, the next words are independent with the pass words
  - Probabilities depend on only the last k words (i.e. the words nearby), instead of long dependency.
  - Only relying on the nearby words to predict the upcomming word

- Perplexity: to evaludate the LM
  - The smaller the perplexity on test sentences, the better the LM is 
- RNN: a better way to build a LM
  - Core idea: share weights at different timestamps 


## RNN-LM for Text Generation
- Greedy Decoding: One to Many network
  - At each timestamp: pick words with the highest prob (greedy)
- Beam Search Decoding:
  - Naïve idea: Pick up the most possible sequence
    - At each time stamp: try all the possible words in the vocab  
  - Beam search decoding: Keep track of k most possible sequence
    - At each timestamp: pick up k words (k= 5~10 for beam size) with highest prob 
      - `k` Beam size: at each timestample, how many words you want to pick 
      - `q` How many paths you want to back-track

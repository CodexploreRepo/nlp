# Natural Language Processing


- In the **word2vec** architecture, the two algorithm names: 
  - `continuous bag of words` ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - `skip-gram` (SG) - **easier to implement**: to find the most related words (context words) for a given center word
- in the **doc2vec** architecture, the corresponding algorithms:
  - `distributed memory` (DM) 
  - `distributed bag of words` (DBOW): this doc2vec model analogous to **Skip-gram model in word2vec**. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.


# BERT - Bidirectional Encoder Representations from Transformers 
- BERT is a departure from the LSTM-based approaches to NLP
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/158507772-3cceec68-d1c4-4b1a-b589-1bc87f4ae740.png" width="400" />
</p>


- BERT Resources
  - [BERT Research - Key Concepts & Sources](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/) 

# Natural Language Processing


- In the **word2vec** architecture, the two algorithm names: 
  - “continuous bag of words” ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - “skip-gram” (SG) - **Easier to implement**: to find the most related words (context words) for a given center word
- in the **doc2vec** architecture, the corresponding algorithms:
  - “distributed memory” (DM) 
  - “distributed bag of words” (DBOW): this doc2vec model analogous to **Skip-gram model in word2vec**. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.

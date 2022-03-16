# Natural Language Processing

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Word2Vec vs Doc2Vec](#1-word2vec-vs-doc2vec)
- [2. BERT](#2-bert)
  - [2.1. BERT Introduction](#21-bert-introduction) 
  - [2.2. BERT Vocab](#22-bert-vocab)


# 1. Word2Vec vs Doc2Vec
- In the **word2vec** architecture, the two algorithm names: 
  - `continuous bag of words` ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - `skip-gram` (SG) - **easier to implement**: to find the most related words (context words) for a given center word
- in the **doc2vec** architecture, the corresponding algorithms:
  - `distributed memory` (DM) 
  - `distributed bag of words` (DBOW): this doc2vec model analogous to **Skip-gram model in word2vec**. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.

[(Back to top)](#table-of-contents)

# 2. BERT
## 2.1. BERT Introduction
- BERT is a departure from the LSTM-based approaches to NLP
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/158507772-3cceec68-d1c4-4b1a-b589-1bc87f4ae740.png" width="400" />
</p>

- BERT Installation via the [Huggingface](https://huggingface.co/docs/transformers/model_doc/bert) implementation: `pip install pytorch-pretrained-bert`
- BERT Resources
  - [BERT Research - Key Concepts & Sources](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/) 

## 2.2. BERT Vocab
- BERT is pre-trained &#8594; Vocab is fixed
- For unkown words (the words that not exist in BERT's vocab), BERT will use [WordPiece Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers) to break the word into the known sub-words
  - **Note 1**: all Sub-words will start with `##`, except the first sub-word in the word (See below table)
  - **Note 2**: By splitting words into word pieces, we have already identified that the words "surfboard" and "snowboard" share meaning through the wordpiece "##board".
    - &#8594; [WordPiece Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers) allows BERT to easily identify related words as they will usually share some of the same input tokens, which are then fed into the first layers of BERT.

<div align="center">
  
| Word          | Token(s)                           |
| ------------- | ---------------------------------- |
| surf          | \['surf'\]                         |
| surfing       | \['surf', '##ing'\]                 |
| surfboarding  | \['surf', '##board', '##ing'\]       |
| surfboard     | \['surf', '##board'\]               |
| snowboard     | \['snow', '##board'\]               |
| snowboarding  | \['snow', '##board', '##ing'\]       |
| snow          | \['snow'\]                         |
| snowing       | \['snow', '##ing'\]                 |

</div>

[(Back to top)](#table-of-contents)

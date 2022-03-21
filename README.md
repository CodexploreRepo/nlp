# Natural Language Processing

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Word2Vec vs Doc2Vec](#1-word2vec-vs-doc2vec)
  - [1.1. Word2Vec](#11-word2vec) 
  - [1.2. Doc2Vec](#12-doc2vec)
- [2. BERT](#2-bert)
  - [2.1. BERT Introduction](#21-bert-introduction) 
  - [2.2. BERT Vocab](#22-bert-vocab)


# 1. Word2Vec vs Doc2Vec
## 1.1. Word2Vec
- In the **word2vec** architecture, the two algorithm names: 
  - `continuous bag of words` ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - `skip-gram` (SG) - **easier to implement**: to find the most related words (context words) for a given center word

<p align="center">
<img width="591" alt="Screenshot 2022-03-20 at 23 39 52" src="https://user-images.githubusercontent.com/64508435/159170384-3f412eb4-3dd5-4618-9bef-4cfd1b3aced3.png"></p>

### 1.1.1. Word2Vec Implementation in Python
- Gensim library in Python will enable us to develop word embeddings by training our own word2vec models on a custom corpus either with CBOW of skip-grams algorithms.
- Training Input format: **a list of lists of tokens**
  - `sentences= [[tokens of sent 1], [tokens of sent2], [tokens of sent ]]=[["sent", "one"], ["sent", "two", "tokens"], ["three"]]`
  - **Note**: Stopwords can be useful to undersand the semantics of the sentence. **Therefore stopwords are not removed while creating the word2vec model**.
```Python
# Initializing the train model
from gensim.models import word2vec

# Creating the model and setting values for the various parameters
num_features = 100  # Word vector dimensionality, can change if word2vec is not working well
min_word_count = 40 # Minimum word count (words with occurrence less than this count will be ignored)
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words: prob to select the negative words
sg = 1              # The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW

print("Training model....")
model = word2vec.Word2Vec(sentences,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling,
                          sg=sg)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model_path = f"/content/drive/MyDrive/Courses/CS601-Introduction to AI/NLPCodes/Word2Vec_Movie_Review/model/{model_name}"
model.save(model_path)
print("model saved")

```


## 1.2. Doc2Vec

- In the **doc2vec** architecture, the corresponding algorithms:
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

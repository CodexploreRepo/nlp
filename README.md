# Natural Language Processing

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Word2Vec vs Doc2Vec](#1-word2vec-vs-doc2vec)
  - [1.1. Word2Vec](#11-word2vec) 
  - [1.2. Doc2Vec](#12-doc2vec)
- [2. BERT](#2-bert)
  - [2.1. BERT Introduction](#21-bert-introduction)
    - [2.1.1. What is BERT ?](#211-what-is-bert)
    - [2.1.2. Why BERT ?](#212-why-bert)
    - [2.1.3. BERT Installation](#213-bert-installation)  
  - [2.2. BERT Vocab](#22-bert-vocab)
  - [2.3. BERT Input and Output](#23-bert-input-and-output)
    - [2.3.1. BERT Input](#231-bert-input)
    - [2.3.2. BERT Output](#232-bert-output) 
- [Resources](#resources)
# 1. Word2Vec vs Doc2Vec
## 1.1. Word2Vec
- In the **word2vec** architecture, the two algorithm names: 
  - `continuous bag of words` ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - `skip-gram` (SG): **easier to implement**: to find the most related words (context words) for a given center word

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
                          sg=
                          )

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model_path = f"/content/drive/MyDrive/Courses/CS601-Introduction to AI/NLPCodes/Word2Vec_Movie_Review/model/{model_name}"
model.save(model_path)
print("model saved")

```
- Load Model & access Vocabs
```Python
model = word2vec.Word2Vec.load(model_path)
print(len(model.wv.vocab))

#print the words in 
for word, vocab_obj in model.wv.vocab.items():
  print(word)
```

## 1.2. Doc2Vec

- In the **doc2vec** architecture, the corresponding algorithms:
  - `distributed memory` (DM) 
  - `distributed bag of words` (DBOW): this doc2vec model analogous to **Skip-gram model in word2vec**. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.





[(Back to top)](#table-of-contents)

# 2. BERT
## 2.1. BERT Introduction
### 2.1.1. What is BERT
- BERT is an acronym for Bidirectional Encoder Representations from Transformers.
- BERT is a departure from the LSTM-based approaches to NLP
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/158507772-3cceec68-d1c4-4b1a-b589-1bc87f4ae740.png" width="400" />
</p>

- BERT architecture consists of several **Transformer encoders** stacked together. 
  - Each Transformer encoder encapsulates two sub-layers: 
    - A self-attention layer 
    - A feed-forward layer.
 
- There are two different BERT models:
  - BERT **base**, which is a BERT model consists of 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.
  - BERT **large**, which is a BERT model consists of 24 layers of Transformer encoder,16 attention heads, 1024 hidden size, and 340 parameters.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/159828395-014a36f8-e459-4a2f-826e-49a38deb3d31.png" width="400" />
</p>

### 2.1.2. Why BERT
There are at least two reasons why BERT is a powerful language model:
1. It is pre-trained on unlabeled data extracted from BooksCorpus, which has 800M words, and from Wikipedia, which has 2,500M words.
2. As the name suggests, it is pre-trained by utilizing the bidirectional nature of the encoder stacks. This means that BERT learns information from a sequence of words not only from left to right, but also from right to left.


### 2.1.3. BERT Installation
- BERT Installation via the [Huggingface](https://huggingface.co/docs/transformers/model_doc/bert) implementation: `pip install transformers`



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

## 2.3. BERT Input and Output
### 2.3.1. BERT Input
BERT model expects a sequence of tokens (words) as an input. In each sequence of tokens, there are two special tokens that BERT would expect as an input:
1. `[CLS]`: This is the first token of every sequence, which stands for classification token.
2. `[SEP]`: This is the token that makes BERT know which token belongs to which sequence. This special token is mainly important for a next sentence prediction task or question-answering task. If we only have one sequence, then this token will be appended to the end of the sequence.
3. `[PAD]`: If the tokens in a sequence are less than `max_length`, say in below example `max_length=10`,we can use padding to fill the unused token slots with `[PAD]` token.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/159833009-1cba88a4-da96-4eec-ac82-245e44fa3f91.png" width="400" />
</p>


- `'bert-base-cased'`: For datasets from different languages, you might want to use `bert-base-multilingual-cased`. 
  - Specifically, if your dataset is in German, Dutch, Chinese, Japanese, or Finnish, you might want to use a tokenizer pre-trained specifically in these languages.
- Maximum size of tokens that can be fed into BERT model is **512**. 
  - If the tokens in a sequence are longer than 512, then we need to do a **truncation**.
- `attention_mask`: which is a binary mask that identifies whether a token is a real word or just padding. If the token contains `[CLS]`, `[SEP]`, or any real word, then the mask would be 1. Meanwhile, if the token is just padding or `[PAD]`, then the mask would be 0.
- `token_type_ids`: binary mask that identifies in which sequence a token belongs. If we only have a single sequence, then all of the token type ids will be 0. For a text classification task, `token_type_ids` is an optional input for our BERT model.


```Python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_test = 'I will watch Memento tonight'
bert_input = tokenizer(
            example_test,
            padding='max_length',
            max_length = 10,
            truncation = True,
            return_tensors='pt'
)

print(bert_input['input_ids'])
print(bert_input['token_type_ids']) #token_type_ids , which is a binary mask that identifies in which sequence a token belongs. If we only have a single sequence, then all of the token type ids will be 0.
print(bert_input['attention_mask']) #attention_mask , which is a binary mask that identifies whether a token is a real word or just padding. 

#tensor([[  101,   146,  1209,  2824,  2508, 26173,  3568,   102,     0,     0]]) - input_ids
#tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) - token_type_ids
#tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]) - attention_mask
```
### 2.3.2. BERT Output
- BERT model then will output an **embedding vector of size 768** in *each of the tokens*.
- These vectors as an input for different kinds of NLP applications: text classification, next sentence prediction, Named-Entity-Recognition (NER), or question-answering.
  - For example:  a text classification task, we focus our attention on the embedding vector output from the special `[CLS]` token. This means that we’re going to use the embedding vector of size 768 from `[CLS]` token as an input for our classifier, which then will output a vector of size the number of classes in our classification task.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/159830486-08bff517-da7b-4980-9f4e-a32313eb6564.png" width="400" />
</p>



[(Back to top)](#table-of-contents)

# Resources:
## Reading List:
### BERT Resources
- [BERT Research - Key Concepts & Sources](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/) 
## To Do:
- [Word Embedding Visualization](https://github.com/marcellusruben/Word_Embedding_Visualization)

[(Back to top)](#table-of-contents)

# Natural Language Processing

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Word (Document) Embedding](#1-word-embedding)
  - [1.1. Classical Document Embedding Techniques](#11-classical-document-embedding-techniques)
    - [1.1.1. Bag of Words](#111-bag-of-words) 
    - [1.1.2. Latent Dirichlet allocation (LDA)](#112-latent-dirichlet-allocation)
  - [1.2. Unsupervised Document Embedding Techniques](#12-unsupervised-document-embedding-techniques)
    - [1.2.1. Word2Vec](#121-word2vec)
    - [1.2.2. Averaging Word Embeddings](#122-averaging-word-embeddings)
    - [1.2.3. Paragraph vectors (doc2vec)](#123-doc2vec)
- [2. BERT](#2-bert)
  - [2.1. BERT Introduction](#21-bert-introduction)
    - [2.1.1. What is BERT ?](#211-what-is-bert)
    - [2.1.2. Why BERT ?](#212-why-bert)
    - [2.1.3. BERT Installation](#213-bert-installation)  
  - [2.2. BERT Vocab](#22-bert-vocab)
  - [2.3. BERT Input and Output](#23-bert-input-and-output)
    - [2.3.1. BERT Input](#231-bert-input)
    - [2.3.2. BERT Output](#232-bert-output) 
- [3. Word2Vec vs Doc2Vec Implementation](#3-word2vec-vs-doc2vec-implementation)
  - [3.1. Word2Vec](#31-word2vec) 
  - [3.2. Doc2Vec](#32-doc2vec)
- [Resources](#resources)

# 1. Word-Embedding
- **Word (Document) embedding** — the mapping of words into numerical vector spaces — has proved to be an incredibly important method for natural language processing (NLP) tasks in recent years, enabling various machine learning models that rely on vector representation as input to enjoy richer representations of text input.
- **Document**: refer to any sequence of words, ranging from sentences and paragraphs through social media posts all way up to articles, books and more complexly structured text documents (e.g. forms)
## 1.1. Classical Document Embedding Techniques:
There are 2 established techniques for document embedding: bag-of-words and latent Dirichlet allocation
### 1.1.1. Bag of Words
#### `Bag of words` Definition
- **Bag of words**: One of the basic models that you should always try with a classification problem in NLP. 
- In bag of words, we create a huge sparse matrix that stores counts of all the words in our corpus (corpus = all the documents = all the sentences)
- *Bag of words* can be implemented in *Sklearn* using `Count Vectorizer`.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/160330243-957a9f75-89a4-4a7a-9bfa-04a54f94f728.png" width="400" />
  <br>A bag-of-words representation of an example sentence
</p>

#### Count Vectorizer
- This is done by deciding on a set of n words that will form the vocabulary supported by the mapping, and assigning each word in the vocabulary a unique index. 
- Then, each document is represented by a vector of length n, in which the i-th entry contains the number of occurrences of the word i in the document.

#### TF-IDF Weighting
- **Term Frequency–Inverse Document Frequency**: re-weights the above word (or n-gram) frequency vectors with the inverse document frequency (IDF) of each word.
  - `TF` term grows as the word appears more often
  - `IDF` term increases with the word’s rarity or how common or rare a word appear in the entire document set.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/160331600-b6ba54b2-d1ca-428b-b16f-41620bf2f60a.png" width="400" />
</p>


#### N-grams
- **N-grams** are combinations of words in order. 
- Downside of this approach is the non-linear dependency of the vocabulary size on the number of unique words, which can be very large for large corpora. Filtering techniques are commonly used to reduce the vocabulary size.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/160330628-6983051b-8d91-4115-87c6-fd9bf72b51ca.png" width="400" />
  <br>1-gram & 2-grams representation of the sentence “The movie is amazing”
</p>

- Both `CountVectorizer` and `TfidfVectorizer` implementations of *scikit-learn* offers ngrams by 
  - `ngram_range` parameter:  which has a minimum and maximum limit. 
    - By default, this is (1, 1). When change it to (1, 3), we are looking at unigrams, bigrams  and trigrams. 
```Python
tfidf_vec = TfidfVectorizer(
      tokenizer=word_tokenize,
      token_pattern=None,
      ngram_range=(1, 3)
)
```

### 1.1.2. Latent Dirichlet allocation

[(Back to top)](#table-of-contents)

## 1.2. Unsupervised Document Embedding Techniques
- **Distributional hypothesis** in linguistics is derived from the semantic theory of language usage, i.e. words that are used and occur in the same contexts tend to purport similar meanings. The underlying idea that *“a word is characterized by the company it keeps”* was popularized by Firth. The distributional hypothesis is the basis for `statistical semantics`.
### 1.2.1. Word2Vec
- In the **word2vec** architecture, the two algorithm names: 
  - `continuous bag of words` ([CBOW](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)):  to predict the current target word (the center word) based on the source context words (surrounding words)
  - `skip-gram` (SG): **easier to implement**: to find the most related words (context words) for a given center word

<p align="center">
<img width="591" alt="Screenshot 2022-03-20 at 23 39 52" src="https://user-images.githubusercontent.com/64508435/159170384-3f412eb4-3dd5-4618-9bef-4cfd1b3aced3.png"></p>

### 1.2.2. Averaging Word Embeddings
- Averaging Word2Vec vectors of words in each document to produce a embedding vector for the document.
### 1.2.3. Doc2Vec
#### Paragraph Vectors: Distributed Memory (PV-DM)
The PV-DM model augments the standard encoder-decoder model by adding a memory vector, aimed at capturing the topic of the paragraph, or context from the input. The training task here is quite similar to that of continuous bag of words; a single word is to be predicted from its context. In this case, the context words are the preceding words, not the surrounding words, as is the paragraph.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/160338673-dfce8bd9-cf35-4f9f-a1e5-094ad6cf3c1b.png" width="400" />
  <br>The Distributed Memory model of Paragraph Vectors (PV-DM)
</p>

#### Paragraph Vectors: Distributed Bag of Words (PV-DBOW)
The second variant of paragraph vectors, despite its name, is perhaps the parallel of word2vec’s skip-gram architecture; the classification task is to predict a single context word using only the paragraph vector. At each iteration of stochastic gradient descent, a text window is sampled, then a single random word is sampled from that window, forming the below classification task.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/160339075-380361f7-39f0-4357-82a8-a2e96d6d0698.png" width="400" />
  <br>The Distributed Bag of Words model of Paragraph Vectors (PV-DBOW)
</p>


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

>>> tensor([[  101,   146,  1209,  2824,  2508, 26173,  3568,   102,     0,     0]]) #input_ids
>>> tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])                                         #token_type_ids
>>> tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])                                         #attention_mask
```
- We also can decode the `input_ids` back to the real words

```Python
example_text = tokenizer.decode(bert_input.input_ids[0])

print(example_text)
>>>'[CLS] I will watch Memento tonight [SEP] [PAD] [PAD]'
```

### 2.3.2. BERT Output
- BERT model then will output an **embedding vector of size 768** in *each of the tokens*.
- These vectors as an input for different kinds of NLP applications: text classification, next sentence prediction, Named-Entity-Recognition (NER), or question-answering.
  - For example:  a text classification task, we focus our attention on the embedding vector output from the special `[CLS]` token. This means that we’re going to use the embedding vector of size 768 from `[CLS]` token as an input for our classifier, which then will output a vector of size the number of classes in our classification task.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/159830486-08bff517-da7b-4980-9f4e-a32313eb6564.png" width="400" />
</p>



[(Back to top)](#table-of-contents)


# 3. Word2Vec vs Doc2Vec Implementation
## 3.1. Word2Vec
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
                          vector_size=num_features,\
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

## 3.2. Doc2Vec

- In the **doc2vec** architecture, the corresponding algorithms:
  - `distributed memory` (DM) 
  - `distributed bag of words` (DBOW): this doc2vec model analogous to **Skip-gram model in word2vec**. The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.






# Resources
## Reading List
- [Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
### BERT Resources
- [BERT Research - Key Concepts & Sources](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/)
- [Multi-label Text Classification with BERT using Pytorch](https://kyawkhaung.medium.com/multi-label-text-classification-with-bert-using-pytorch-47011a7313b9)
## Todo List
- [Word Embedding Visualization](https://github.com/marcellusruben/Word_Embedding_Visualization)
- [Spaces: How to Showcase Your ML Web App Demo in Public](https://towardsdatascience.com/spaces-how-to-showcase-your-ml-web-app-demo-in-public-3a701772959)

[(Back to top)](#table-of-contents)

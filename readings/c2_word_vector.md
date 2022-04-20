- Example: Textual Similarity
  - Given 2 sentences, we want to determine 2 sentences are similar or not 
  <img width="1065" alt="Screenshot 2022-04-20 at 19 11 13" src="https://user-images.githubusercontent.com/64508435/164218294-72d8da77-2d9e-4bd3-81c9-571a2838b19c.png">
  - Problem: Similar meanings but different words
- [1. Corpus Pre-processing](#1-corpus-pre-processing)


# 1. Corpus Pre-processing
## 1.1. Regular Expression
- Definition:
  - Search *pattern* used to match character combinations in a string
  - Pattern: a sequence of characters
- Application:
  - Search and replace
  - Validate texts to ensure given patterns (for example: password validation)
  - <img width="607" alt="Screenshot 2022-04-20 at 19 16 34" src="https://user-images.githubusercontent.com/64508435/164219135-c46cd292-0357-43fa-b905-18573e7e2863.png">

- Fixed pattern: **Exact** match
- Meta characters: **Relaxed** match a type of characters
<img width="405" alt="Screenshot 2022-04-20 at 19 16 18" src="https://user-images.githubusercontent.com/64508435/164219098-30b155b6-7cb5-4c81-b40b-527571855b43.png">

- Character class:<img width="675" alt="Screenshot 2022-04-20 at 19 20 44" src="https://user-images.githubusercontent.com/64508435/164219763-c610452a-937b-4213-9879-a1079485f99a.png">

- Repetition: repeated match<img width="514" alt="Screenshot 2022-04-20 at 19 18 14" src="https://user-images.githubusercontent.com/64508435/164219352-9a354e34-c9a0-4563-8878-cd453f4c7f63.png">

- Example: Regular Expression
- <img width="607" alt="Screenshot 2022-04-20 at 19 19 55" src="https://user-images.githubusercontent.com/64508435/164219626-897d01b9-fd96-4544-ac78-071a4df8ebbf.png">
- Group: 
- Enclosing with “( ... )”
- Match the whole pattern, while capture each group individually 
- Groups can be nested “( ... (... (...))...)”

<img width="545" alt="Screenshot 2022-04-20 at 19 22 05" src="https://user-images.githubusercontent.com/64508435/164219973-e1c41737-0121-41d8-98ae-3b1f5776a708.png">

- Backreference: Find repeated group
<img width="601" alt="Screenshot 2022-04-20 at 19 22 36" src="https://user-images.githubusercontent.com/64508435/164220044-16e3df75-411a-4367-af5a-d27542c6f41b.png">
  - `\2`: back-reference the second group
  - Matched words: `m-o-m`, `t-e-s-t`

- Group assertions: Capture ahead or behind the match
<img width="601" alt="Screenshot 2022-04-20 at 19 25 50" src="https://user-images.githubusercontent.com/64508435/164220488-9906e317-4425-46e6-bda2-693f6aa0eff9.png">

## 1.2. Tokenization
- **Tokenization**: Split a piece of text into meaningful units (i.e., tokens)
- **Token**: word, sub-word (like morphemes), numbers, or punctuation, etc.

- Important steps for word representation
  – Especially for the languages without word spacing (like Chinese)
- Method 1: Regular Expression
  - Lightweight, efficient in practice
- Method 2: Using packages, e.g., spaCy
  - Limitation: languages with whitespaces separating words   
<img width="699" alt="Screenshot 2022-04-20 at 19 32 54" src="https://user-images.githubusercontent.com/64508435/164221552-efeb7322-e867-405a-b2f3-ee22d3da23ef.png">

### 1.2.2. Tokenization for languages without separate whitespaces
- Tokenization for languages without separate whitespaces ?
- Intuitive solution
  – First collect a dictionary of words 
  - Then conduct Maximum Matching
- Limitation: perform worse on languages with whitespaces
<img width="713" alt="Screenshot 2022-04-20 at 19 34 30" src="https://user-images.githubusercontent.com/64508435/164221795-ee4ce6ba-6ab1-4e2f-a56a-eb07be464408.png">

<img width="480" alt="Screenshot 2022-04-20 at 19 35 00" src="https://user-images.githubusercontent.com/64508435/164221871-4d015b5f-6d13-4bc7-8bd0-8da08e1a445f.png">

### 1.2.3. Tokenization using Sub-word
- Tokenization Sub-word: Split texts into frequent subwords
- Current state-of-the-art:
  - **Byte-Pair Encoding (BPE)**: GPT, Roberta, XLM 
  - **WordPiece**: BERT, DistilBERT, Electra
#### 1.2.3.1. Tokenization: BPE
- Two steps:
  - Token learning: learn a vocabulary from corpus 
  - Tokenizer: split text according to the vocabulary

<img width="625" alt="Screenshot 2022-04-20 at 19 40 29" src="https://user-images.githubusercontent.com/64508435/164222677-05cbda66-0921-407e-8c25-b9949359abd1.png">
 - Step 1: Token Learning
  - `k`: how many sub-words that we want 
  - Initialize the vocab with single character
  - Find most frequent two consecutive.
  - By Product: Merge Rules
<img width="643" alt="Screenshot 2022-04-20 at 19 41 52" src="https://user-images.githubusercontent.com/64508435/164222889-04c3a6ad-6beb-4d0c-a9b6-bb652de93e7f.png">
    - Find most frequent two consecutive. In this example is `es`
    <img width="681" alt="Screenshot 2022-04-20 at 19 43 48" src="https://user-images.githubusercontent.com/64508435/164223175-f0d5e6cc-2cb5-4c83-a88e-c032788b400f.png">
<img width="681" alt="Screenshot 2022-04-20 at 19 44 06" src="https://user-images.githubusercontent.com/64508435/164223217-521eac6d-0397-4dc6-8868-350aeca3cdb8.png">

- Step 2: Tokenier
  - Base on the **merge rules**, we will build the tokenizer. 
<img width="594" alt="Screenshot 2022-04-20 at 19 45 38" src="https://user-images.githubusercontent.com/64508435/164223468-7d45b76e-21a1-4cf9-a9ec-92ff2b665d6c.png">

#### 1.2.3.2. Tokenization: WordPiece
- Mostly, similar to BPE
- WordPiece does not choose the most frequent pair, but the one that **maximizes the likelihood of the training data once added to the vocabulary**.
  - evaluates what it loses by merging two symbols to ensure it’s worth it.
  
<img width="685" alt="Screenshot 2022-04-20 at 19 54 55" src="https://user-images.githubusercontent.com/64508435/164224944-35b95677-0c21-446b-977b-49c296ab1d8f.png">

## 1.3. Normalization
- Convert the meaningful unit into a canonical (one) form
  - Lemma, Stem
<img width="664" alt="Screenshot 2022-04-20 at 19 57 18" src="https://user-images.githubusercontent.com/64508435/164225282-1f9ca9c6-3a4a-4eba-87a6-01f7fe399187.png">

- Not always necessary 
  - helpful in Information Retrieval
  - may bring noise in Named Entity Recognition

<img width="664" alt="Screenshot 2022-04-20 at 19 58 14" src="https://user-images.githubusercontent.com/64508435/164225460-6ed72aff-603a-4846-8251-1f4e0bcfba09.png">

### 1.3.1. Normalization - Stemming
- Idea: Reduce words to their stem (stem: meaningful unit in a word)
• Method: removing affixes based on rules – Singular vs plural
– verb tenses
– comparative/superlative adjective
• Pro: fast
• Con: Stem word may not be a word

<img width="312" alt="Screenshot 2022-04-20 at 19 59 37" src="https://user-images.githubusercontent.com/64508435/164225691-e96bd3a1-453f-48ef-a19a-bd29af6de2ca.png">

- Porter Stemmer:  most common stemmer for English
  - Series of rules that run in a cascade:
    - output of each pass is fed into the next pass 
    - stemming stops if a pass makes no changes

<img width="487" alt="Screenshot 2022-04-20 at 20 01 22" src="https://user-images.githubusercontent.com/64508435/164226008-379102b3-29ab-40cf-87a5-5acbe809fef6.png">

### 1.3.2. Normalization - Lemmatization
- Idea: convert words to the base form
- Lemma (Base form) will depend on the POS (Part of Speed) tage of a word like Lemma (n), Lemma (v)

- Pro:
  – Lemma words are proper words
  - Can normalize irregular forms
- Con:
  – require lexicon + rules – require POS tags
  - slower than stemming as need to identify POS tags
<img width="542" alt="Screenshot 2022-04-20 at 20 02 46" src="https://user-images.githubusercontent.com/64508435/164226217-682e3a2f-43ab-40c3-a1fd-d03b00c6eda3.png">

#### Final words for Normalization:
- Canonical form also effects tokenization
  - Separate out clitics (e.g., doesn’t -> does n’t, John’s -> John ‘s) – Keep hyphenated words together
  - Separate out all punctuation symbols
- Other common normalization steps
  - Removal of stop words (e.g, a, an, the)
  - Removal of non-standard tokens (e.g., url, emojis)


# 2. Word Vector
- Motivation: Find the similar words
## 2.1. Sparse Word Embeddings
### 2.1.1. One-hot Encoding 
<img width="677" alt="Screenshot 2022-04-20 at 20 11 07" src="https://user-images.githubusercontent.com/64508435/164227554-089f9e07-52d5-46dd-a2e9-081e93b976db.png">

### 2.1.2. Word Net
- WordNet is a lexical knowledge base 
  - mostly **manually** curated;
  – gather similar words into synsets
  - `synsets` are connected via various relations

<img width="689" alt="Screenshot 2022-04-20 at 20 14 44" src="https://user-images.githubusercontent.com/64508435/164228136-0ee7934c-5874-4c64-bc9b-cd023d294308.png">

- WordNet based Word similarity (e.g., Wu & Palmer Similarity) 
  - Topological similarity between synsets
  - Consider the depth of Least Common Subsumer (LCS)
![IMG_AE8EC93AAAA2-1](https://user-images.githubusercontent.com/64508435/164228835-8c53aa27-1f27-466c-a73a-547d66f55dd5.jpeg)
![IMG_22263D3C34B7-1](https://user-images.githubusercontent.com/64508435/164228928-35e73c1b-ca33-46b5-a9fe-56a8bea99b76.jpeg)
<img width="700" alt="Screenshot 2022-04-20 at 20 20 58" src="https://user-images.githubusercontent.com/64508435/164229159-210c2aee-b016-4fff-bbcc-d5efc607cb74.png">

- Limitation of WordNet similarity
  - Granularity was not fine enough
    - e.g. good vs. effective, charter vs. lease
  - Difficult to keep up-to-date
    - e.g. cloud computing, badass, COVID
  - Human labor intensive

### 2.1.3. Co-Occurrence Vectors
– A vector has a length of vocabulary size (number of unique words)
– Each element denotes how often two words occurs with each other (within a predefined window).

<img width="716" alt="Screenshot 2022-04-20 at 20 23 50" src="https://user-images.githubusercontent.com/64508435/164229612-2cdddbd8-0d1f-4ffe-bd71-955205abe4e3.png">
- In this example: we count the occurence in the entire sentence.
<img width="716" alt="Screenshot 2022-04-20 at 20 25 58" src="https://user-images.githubusercontent.com/64508435/164229926-21c88344-35af-4796-b2da-9b252f49ecf2.png">

- Consideration #1: Relative importance 
  - e.g., is “crowd” twice important than “work”?
- Consideration #2: Discriminative
  - e.g., some words are just very frequent

Solution: from raw count to **Pointwise Mutual Information** (PMI)

#### Pointwise Mutual Information (PMI)
- `p(w1)`, `p(w2)`: probability of unigram, or probability of the word w1 appears in the corpus.
- `pmi(w1, w2)`: whether two words w1, and w2 appear together more freq than if they appear alone. 
- <img width="428" alt="Screenshot 2022-04-20 at 20 50 32" src="https://user-images.githubusercontent.com/64508435/164234315-27c8bfee-09a7-4371-b606-2563e4b0b904.png">

##### Information Theory
- When x occurs, how much information it brings, aka, surprisal of x.
- Inversely proportional to probability
<img width="405" alt="Screenshot 2022-04-20 at 20 55 51" src="https://user-images.githubusercontent.com/64508435/164235290-23079be5-4277-4634-8631-05104cda4315.png">
- Example 1:
<img width="640" alt="Screenshot 2022-04-20 at 20 56 32" src="https://user-images.githubusercontent.com/64508435/164235424-48bc4bfb-f27c-494e-a8df-389afcd737a1.png">
- Example 2: danger brings more information than "the", "an", "and"

<img width="640" alt="Screenshot 2022-04-20 at 20 57 36" src="https://user-images.githubusercontent.com/64508435/164235611-329fde8d-8730-43bd-9838-c05d6eed48b5.png">

##### Entropy
- Expectation of information contents
- More uncertainty, higher entropy
- High Entropy, the system will be un-stable as there is a lot of uncertainty.

<img width="395" alt="Screenshot 2022-04-20 at 20 59 29" src="https://user-images.githubusercontent.com/64508435/164235916-7961bbfa-1059-4581-897e-0634554ab676.png">


##### Mutual Information
- Mutual Information: Entropy betweeen two systems
- If X & Y independent, then I(X,Y) = 0
- I see the word "movie", it will have me to guess the next words, for example "star", "actor"
- Mutual information is the average level of PMI.
<img width="678" alt="Screenshot 2022-04-20 at 21 01 36" src="https://user-images.githubusercontent.com/64508435/164236277-36f5e654-dc74-45cd-a4bd-b5f085d23984.png">

### Converting Count-based Matrix to PMI Matrix

![IMG_AC5E0FDC50AE-1](https://user-images.githubusercontent.com/64508435/164237310-850046c4-6988-4960-b67d-9377b9a9ab48.jpeg)

#### Limitation of 
– Vector size is too large 1 x |𝑉| , (|𝑉| is typically 20k~200k)
– Vector is too sparse (most dimensions are zero value) 
  - Various improvements, e.g., Add-1 smoothing


## 2.2. Dense Word Embeddings
- A dense vector (a.k.a.  word embeddings) that maps the similarity between words to the distance in vector space.

<img width="695" alt="Screenshot 2022-04-20 at 21 10 18" src="https://user-images.githubusercontent.com/64508435/164237724-78f880de-c540-42be-8ec6-b4b52ea6cd19.png">

### 2.2.1. Why Dense Word Vectors


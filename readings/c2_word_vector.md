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



# Naive Bayes

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Probability](#1-probability)
  - [1.1. Introduction to Prob](#11-introduction-to-prob)
  - [1.2. Conditional Prob](#12-conditional-prob)
  - [1.3. Bayes' Rule](#13-bayes-rule)
- [2. Naïve Bayes](#2-naive-bayes)
  - [2.1. Laplacian Smoothing](#21-laplacian-smoothing) 

# 1. Probability
## 1.1. Introduction to Prob
- `N`: Total number of tweet (In this example is 20)
- `N_pos`: Total number of positive tweet
- `P(A)` = prob of positive tweets 
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161574869-a7afc8a4-2aae-47ab-b0b6-936a87e04af6.png" width="600" />
</p>

- `P(B)` = prob of tweets contain the word "happy"
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161575054-21084164-09e2-433d-a7e8-ce56cf032a36.png" width="600" />
</p>

- `P(A and B)` = The prob of positive tweets and tweets contains "Happy"
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161575517-36e321e7-17b8-431e-9138-4f45e56302c0.png" width="600" />
</p>

## 1.2. Conditional Prob
<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/64508435/161575916-0bd09d85-9e24-4dab-b7a2-92cc39d47266.png">
</p>

## 1.3. Bayes' Rule
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161576372-a65836e2-585e-40ef-b9e9-3be721cd322f.png" width="500" />
</p>

[(Back to top)](#table-of-contents)

# 2. Naive Bayes
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161576834-73010d15-089c-432c-b40a-7f82f336458e.png" width="500" />
</p>

- Construct  `P(word|class)` for each word in Vocab.
  - where class = {Pos, Neg}
- Prob of "I", "am", and "NLP" are not significantly different in each class
- Meanwhile "happy", "sad" are significantly different in Positive and Negative classes.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161581186-df4dddaa-fc87-4d29-820b-5573761d0982.png" width="500" />
<img width="600" alt="image" src="https://user-images.githubusercontent.com/64508435/161581437-f9cd0d01-68d6-4d81-aa09-c387eea3d4d5.png">
</p>

- Once you have the probabilities, you can compute the **likelihood score** as follows
  - A score greater than 1 indicates that the class is positive, otherwise it is negative.
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161581729-ca85f794-9e9f-4b94-9b03-ed4b05a5c36f.png" width="500" />
</p>

## 2.1. Laplacian Smoothing
- If a word does not appear in the training, then it automatically gets a probability of 0, to fix this we add **Laplacian smoothing** as follows:
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161584073-e481a14e-629d-4b99-a488-f69eb8d818cf.png" width="400" />
</p>


- Where: 
  - `N_class`: frequency of all words in class
  - `V`: number of unique words in vocabulary
- **Note**: Added a `1` in the numerator, and since there are `V` words to normalize, we add `V`in the denominator, so that the total prob = 1.
- Example: `P(w_i|class)` with smoothing
<p align="center">
<img src="https://user-images.githubusercontent.com/64508435/161582597-9c622eab-db12-45d5-85ef-d9cbc6cfa93f.png" width="500" />
<img src="https://user-images.githubusercontent.com/64508435/161582839-c2a1c875-ef52-43af-b0b2-1a9ceef59c60.png" width="500" />
</p>




[(Back to top)](#table-of-contents)

# Resources
## Reading List

[(Back to top)](#table-of-contents)

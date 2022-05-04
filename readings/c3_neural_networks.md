# Neural Networks


# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction to Machine Learning](#1-introduction-to-machine-learning)
  - [1.1. Reinforcement Learning](#11-reinforcement-learning)
  - [1.2. Unsupervised Learning](#12-unsupervised-learning)
  - [1.3. Supervised Learning](#13-supervised-learning)
  - [1.4. Generative vs Discriminative](#14-generative-vs-discriminative)
  - [1.5. Gradient Descent](#15-gradient-descent)

# 1. Introduction to Machine Learning
- A field that studies **patterns/features** from massive amount of data that can be applied for decision making on *unseen data under uncertainty*.
- Main types of methods:
  - Reinforcement learning
    - learn to make decisions directly when given occasional reward or punishment signals.
  - Unsupervised learning
    - learn patterns/features from data that doesn’t have input-output correspondence.
  - Supervised learning
    - learn patterns/features from data that has input-output correspondence.

## 1.1. Reinforcement Learning
- Basic RL model:
  - A set of **states (environment)**
  - A set of **actions**
  - **Policy**: how to take actions given states – Reward: which actions/states are desired
<p align="center"><img width="300" alt="Screenshot 2022-04-20 at 19 25 50" src="https://user-images.githubusercontent.com/64508435/165506768-173daa91-78b4-4b94-95ab-256728d6f89b.png"></p>

- Example of RL:
  - *AlphaGo*: Plays the board game Go by attempting to match the moves of experts.
  - *AlphaGo Zero*: initially knew nothing about Go beyond the rules. Playing against itself.
  - *AlphaStar*: outputs a sequence of instructions that constitute an action within the game (StarCraft).

## 1.2. Unsupervised Learning
- To generalize patterns/factors out of data which doesn’t have supervised signals.
  - Only input data, no outputs.
  - No labeled data.
  - Sometime called knowledge discovery.

<p align="center"><img width="300" alt="Screenshot 2022-04-20 at 19 25 50" src="https://user-images.githubusercontent.com/64508435/165507997-d7ce1b97-7411-4d21-b6c0-2ab2e230f9d4.jpeg"></p>

- Discovering clusters
  - To cluster data into groups and estimate which cluster each data sample belongs to.
    - e.g., Word sense discovery.
- Discovering latent factors
  - To project the data to a lower dimensional subspace which captures the essence of the data.
    - e.g., PCA, dimension reduction.
- Matrix completion
  - To fill in missing values in a matrix.
    - e.g., collaborative filtering for recommendation.

## 1.3. Supervised Learning
- Supervised Learning: To learn a mapping between inputs and outputs given input- output pairs.
  - Classification 
  - Regression

## 1.4. Generative vs Discriminative
- Generative classifier: model the distribution of individual classes
– e.g., Lean the language itself
– Pros: detect outlier, deal with latent variable, better generalization ability – Cons: need more data and more computation to learn distribution
• Discriminative classifier: learn the boundaries between classes – e.g., know the language difference
– Pros: better classification performance, efficient.
– Cons: overfit
### 1.4.1. Probabilistic Classifier
<p align="center">
<img width="600" alt="Screenshot 2022-04-27 at 19 32 45" src="https://user-images.githubusercontent.com/64508435/165509180-12ddb700-1b07-444b-87ad-2f49626fecd0.png">
<img width="600" alt="Screenshot 2022-04-27 at 19 33 35" src="https://user-images.githubusercontent.com/64508435/165509324-bf612d2b-f3c1-4b98-97d7-e9273ae8fc3a.png">
</p>

### 1.4.2. Generative Classifier
- Model the distribution of individual classes
  - build different models for each class
  - classify new data by comparing it with each class distribution
- Typical generative classifier: Naïve Bayes (NB), HMM, K-means, etc.
- Example: Sentiment analysis using NB

<img width="1105" alt="Screenshot 2022-04-27 at 19 36 29" src="https://user-images.githubusercontent.com/64508435/165509830-f6d179c5-8971-4d03-82d4-f1d4dbb54683.png">
- Bag of Words: we will destroy the order of the words in the paragraph.

<img width="1105" alt="Screenshot 2022-04-27 at 19 37 28" src="https://user-images.githubusercontent.com/64508435/165509980-38b3ea05-60b9-4794-b81e-d41f531f8933.png">
- Prior: you have known nothing about the data, but there are some classes y that are very popular
<img width="1122" alt="Screenshot 2022-04-27 at 19 38 51" src="https://user-images.githubusercontent.com/64508435/165510211-69f50ecc-df71-45be-86ca-9b84f12e8395.png">
<img width="1122" alt="Screenshot 2022-04-27 at 19 40 43" src="https://user-images.githubusercontent.com/64508435/165510499-ccca1a77-a11e-483a-9710-7ca086d75ca7.png">
- Ignore the marignal: p(x) or p(w1,w2, w3, .., wn) as we dont care about the prob of the sentence alone.

<img width="1122" alt="Screenshot 2022-04-27 at 19 42 22" src="https://user-images.githubusercontent.com/64508435/165510755-d8beb36d-10c6-4dc5-b88c-16be8abecfa4.png">

- Risk of arithmetic underflow
  - Compute `log` probability
  - Smoothing since there are unseen words or un-presented classes
<p align="center">
<img width="600" alt="Screenshot 2022-04-27 at 19 44 53" src="https://user-images.githubusercontent.com/64508435/165511214-004b1396-e063-437d-94be-384d25caf9a6.png">
<img width="600" alt="Screenshot 2022-04-27 at 19 45 10" src="https://user-images.githubusercontent.com/64508435/165511250-ddb09ec3-ff2e-4ee5-8f51-ad4b2e737097.png"></p>

- Example of NB
<p align="center"><img width="600" alt="Screenshot 2022-04-27 at 19 45 10" src="https://user-images.githubusercontent.com/64508435/165511661-bfe3879f-abfb-47a2-a973-45dc9de1707c.jpeg"></p>

### 1.4.3. Discriminative Classifier

## 1.5. Gradient Descent
### 1.5.1 Loss Function 
#### 1.5.1.1. Loss Function for Classification
- Cross entropy loss for binary logistic regression
<img width="746" alt="Screenshot 2022-04-27 at 21 00 21" src="https://user-images.githubusercontent.com/64508435/165523605-eefb3a7e-66e4-4137-b231-2261b50c45ef.png">

- Cross entropy loss for multicalss logistic regression
<img width="991" alt="Screenshot 2022-04-27 at 21 00 37" src="https://user-images.githubusercontent.com/64508435/165523661-fc595d46-0798-4a4b-8789-0f78a6a4916d.png">

#### 1.5.1.2. Loss Function for Regression
- Multivariate Linear Regression
<img width="991" alt="Screenshot 2022-04-27 at 21 02 20" src="https://user-images.githubusercontent.com/64508435/165523984-2497f32c-f15b-480a-b4d2-00192d533223.png">

#### 1.5.1.3. Loss Function for Word2Vec
<img width="581" alt="Screenshot 2022-04-27 at 21 03 10" src="https://user-images.githubusercontent.com/64508435/165524118-d87aa87a-af0c-4d73-8075-3d58bb7ab7d8.png">

### 1.5.1. Gradient Descent for Logistic Regression

<img width="1017" alt="Screenshot 2022-04-27 at 21 06 31" src="https://user-images.githubusercontent.com/64508435/165524707-72a6541b-a922-444d-923e-c66f80364c9e.png">
<img width="1094" alt="Screenshot 2022-04-27 at 21 15 05" src="https://user-images.githubusercontent.com/64508435/165526265-34651a78-91ac-48c4-a5af-e922a3304b6e.png">
<img width="1094" alt="Screenshot 2022-04-27 at 21 17 03" src="https://user-images.githubusercontent.com/64508435/165526627-21a20b8b-715a-4bd7-a202-7dbba8857843.png">
<img width="1133" alt="Screenshot 2022-04-27 at 21 19 49" src="https://user-images.githubusercontent.com/64508435/165527202-19c02a76-c954-495a-8532-27fc53cf718e.png">

- `Warmup` technique: do a few small steps before large steps
<img width="833" alt="Screenshot 2022-04-27 at 21 21 38" src="https://user-images.githubusercontent.com/64508435/165527575-f693a54e-beab-499c-bfe7-4c954c593b78.png">

## 1.5. Gradient Descent - Variations
- Basic gradient descent: Update once per full set
- Stochastic gradient descent: Update per each data sample
- Mini-batch gradient descent: Update per a sub-set (Usually use)

- When to stop
<img width="931" alt="Screenshot 2022-04-27 at 21 27 48" src="https://user-images.githubusercontent.com/64508435/165529013-839eb429-12bc-4341-b374-37e6b672a4ec.png">

- Optimizer: 
- **SGD with momentum**: Considering the previous step to current update.
- Nesterov accelerated gradient (NAG): Considering the next step to current update.
- Adagrad: Adapt learning rate to parameters
- Adadelta & RMSprop: Restrict the window of accumulated past gradients to a fixed size 
- **Adam**: stores running average of past squared gradients like Adadelta and RMSprop and consider momentum
  - If data is spare, you can use Adam to adaptive change the learning rate to avoid by trapped to local mimimum.
  - If data is sufficient, you can use SGD, but it might take sometime to convert
<img width="595" alt="Screenshot 2022-04-27 at 21 32 18" src="https://user-images.githubusercontent.com/64508435/165530326-f5d21706-1fea-4080-8ae7-d4aef6b14e2b.png">

# 3. Neural Networks
- Network of neurons
  - Each neuron can be regard as a logistic function 
  - “Stacked” logistic regression
<img width="1113" alt="Screenshot 2022-04-27 at 21 39 12" src="https://user-images.githubusercontent.com/64508435/165531758-95b1c26b-3220-4efb-92c6-19756736cfba.png">

![IMG_3D9D5DCCA480-1](https://user-images.githubusercontent.com/64508435/165532399-dd1a4bcf-bba4-4743-a001-2c1745f24cde.jpeg)

## 3.1. Activation Function
Only requirement: non-linear function (to avoid simply linear combination)
![IMG_6FA1547B6C04-1](https://user-images.githubusercontent.com/64508435/165533203-6df1f2fa-401c-4aa1-9c52-e3fa05603494.jpeg)

## 3.2. Optimization
- Forward-propagation
  - Neural net receives an input and generates an output
  - For training stage, it flows onward to produce a scalar cost L
- Back-propagation
  - gradients with regard to each model parameters from the cost flows backward through the network

<img width="439" alt="Screenshot 2022-04-27 at 21 51 19" src="https://user-images.githubusercontent.com/64508435/165534200-bbc6d3cd-685d-4566-a697-244aa2ebb6cd.png">

### 3.2.1. Chain Rules
- The chain rule: expresses the derivative of the composition of two differentiable functions in terms of their own derivatives
<img width="670" alt="Screenshot 2022-04-27 at 21 54 25" src="https://user-images.githubusercontent.com/64508435/165534926-c98f09b2-04b9-421f-bb6a-3fc356865166.png">
- Multiple-path chain rule

<img width="962" alt="Screenshot 2022-04-27 at 21 54 49" src="https://user-images.githubusercontent.com/64508435/165535000-4be9dc29-145d-4d7c-a27e-4dc80fd01bb7.png">

### 3.2.2. Forward pass
<img width="1117" alt="Screenshot 2022-04-27 at 21 56 05" src="https://user-images.githubusercontent.com/64508435/165535268-0c979115-5f68-4d19-b2e1-62c682640eff.png">

### 3.2.3. Backward pass
- To compute the “contributions” of each weight to the total loss 
  - Partial derivatives w.r.t. each weight
<img width="977" alt="Screenshot 2022-04-27 at 21 58 32" src="https://user-images.githubusercontent.com/64508435/165535782-7be0a5fb-53d2-4bbc-b643-7636c8cdecd9.png">

<img width="900" alt="Screenshot 2022-04-27 at 21 59 08" src="https://user-images.githubusercontent.com/64508435/165535882-bed5f051-6061-4e8b-bbd2-11d997e3613f.png">

[(Back to top)](#table-of-contents)


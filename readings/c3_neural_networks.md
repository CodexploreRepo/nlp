# Neural Networks



# Table of contents
- [Table of contents](#table-of-contents)
- [1. Introduction to Machine Learning](#1-introduction-to-machine-learning)
  - [1.1. Reinforcement Learning](#11-reinforcement-learning)
  - [1.2. Unsupervised Learning](#12-unsupervised-learning)
  - [1.3. Supervised Learning](#13-supervised-learning)

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

[(Back to top)](#table-of-contents)

# Machine-Learning--Deep-Q-Network
## Problem Overview
The main objective of this project is to implement a Reinforcement Learning (RL) Agent using two different approaches:
1. **Tabular Q-Learning**
2. **Deep Q-Network (DQN)** — utilizing a neural network to approximate the Q-function.

The RL Agent is designed to learn how to solve a problem in a given environment by interacting with it and improving its strategy over time.

## Reinforcement Learning Environment
The environment for this project is based on a classic control problem from OpenAI's Gymnasium API: the **Cart Pole** problem, originally described by Barto et al. 

In this scenario:
- A pole is attached to a cart via an un-actuated joint, and the cart moves along a frictionless track.
- The pole starts upright, and the goal is to balance it by applying left or right forces to the cart, preventing the pole from falling.

The challenge is to balance the pole for as long as possible by dynamically applying forces in the correct direction.

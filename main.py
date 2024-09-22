import gymnasium as gym
import numpy as np
import setuptools.dist
import tensorflow as tf
from dqn import DQNAgent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 50
    done = False
    batch_size = 64
    rewards = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                if e % 10 == 0:  # Update target model every 10 episodes
                    agent.update_target_model()
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Total Reward: {total_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay()
        rewards.append(total_reward)

    # Convert rewards to numpy array for linear regression
    rewards = np.array(rewards)
    episodes = np.arange(len(rewards))

    # Perform linear regression
    coef = np.polyfit(episodes, rewards, 1)
    poly1d_fn = np.poly1d(coef)

    # Plot rewards over episodes
    plt.plot(episodes, rewards, 'bo', label='Episode Rewards')  # 'bo' means blue color, round points
    plt.plot(episodes, poly1d_fn(episodes), '--k', label='Trend Line')
    plt.plot(episodes, rewards, label='Exact Reward Line')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Over Episodes')
    plt.legend()
    plt.savefig('deep_q-network_rewards_plot.png')
    plt.show()

    # Evaluation phase
    rewards = []
    for episode in range(10):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.01)  # Minimal exploration during evaluation
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = np.reshape(next_state, [1, state_size])
        print(f"Evaluation Episode: {episode+1}, Reward: {total_reward}")
        rewards.append(total_reward)

    print(f"Average Total Reward (Evaluation): {np.mean(rewards)}")
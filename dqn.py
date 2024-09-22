import numpy as np
import setuptools.dist
import tensorflow as tf
from keras import layers, models, optimizers
from replay_buffer import ReplayBuffer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001, batch_size=64, memory_size=20000):
        #Number of features in the state space
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, epsilon=0.1):
        state = np.reshape(state, [1, self.state_size])  # Ensure state is correctly shaped
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        #Checks if there are enough experiences in the memory to sample
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        states = np.array([experience[0] for experience in minibatch])  #Extracts states from the minibatch.
        next_states = np.array([experience[3] for experience in minibatch])  #Extracts next states from the minibatch.
        states = np.reshape(states, [self.batch_size, self.state_size])
        next_states = np.reshape(next_states, [self.batch_size, self.state_size])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q_values[i]) #Adds the discounted maximum Q-value of the next state to the reward
            targets[i][action] = target
        #model's weights are updated to minimize the MSE loss
        self.model.fit(states, targets, epochs=1, verbose=0)
        self.update_target_model()

    def train(self, env, n_episodes=1000):
        max_steps_per_episode = 200  # Set a maximum number of steps per episode
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        epsilon = epsilon_start

        for e in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            state = np.reshape(state, [1, self.state_size])
            for time in range(max_steps_per_episode):
                action = self.act(state, epsilon)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    logging.info(f"Episode: {e}/{n_episodes}, Score: {time}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
                    break
            #Calls the replay method to train the neural network on experiences stored in the replay buffer.
            # This method updates the Q-values using sampled experiences.
            self.replay()
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

    def evaluate(self, env, n_episodes=10):
        total_rewards = 0
        for e in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            state = np.reshape(state, [1, self.state_size])
            while not done:
                action = np.argmax(self.model.predict(state, verbose=0)[0])
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                state = np.reshape(next_state, [1, self.state_size])
                episode_reward += reward
            total_rewards += episode_reward
            logging.info(f"Evaluation Episode: {e}/{n_episodes}, Reward: {episode_reward}")
        return total_rewards / n_episodes

    def render_episode(self, env):
        state, _ = env.reset()
        done = False
        state = np.reshape(state, [1, self.state_size])
        while not done:
            action = np.argmax(self.model.predict(state, verbose=0)[0])
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = np.reshape(next_state, [1, self.state_size])
            logging.info(env.render())

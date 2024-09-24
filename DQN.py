import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        
        # Define the layers (simplified)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # Activation
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, adjacency_matrix, num_nodes, charging_stations, gamma=0.99, epsilon=0.2, epsilon_decay=0.97, min_epsilon=0.05, batch_size=128, memory_size=50000, learning_rate=0.01, target_update_frequency=50):
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.charging_stations = charging_stations

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_frequency = target_update_frequency

        self.memory = deque(maxlen=memory_size)
        self.model = DQNetwork(self.num_nodes, self.num_nodes)
        self.target_model = DQNetwork(self.num_nodes, self.num_nodes)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.epoch_rewards = []
        self.travel_times = []
        self.step_counter = 0

    def update_target_network(self):
        """Copy weights to the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Stores transitions in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def epsilon_greedy_action(self, state):
        """Epsilon-greedy action selection."""
        current_state_idx = np.argmax(state)  # Extract index of current state
        
        if random.random() < self.epsilon:
            valid_actions = np.where(self.adjacency_matrix[current_state_idx] > 0)[0]
            return random.choice(valid_actions)
        else:
            state = np.array(state)
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.device)
            q_values = self.model(state_tensor).detach().cpu().numpy()[0]
            valid_actions = np.where(self.adjacency_matrix[current_state_idx] > 0)[0]
            return valid_actions[np.argmax(q_values[valid_actions])]

    def train_step(self):
        """Train the model based on experiences from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_update_frequency == 0:
            self.update_target_network()

    def dqn_learning(self, start_state, end_state, num_epoch):
        best_path = []
        best_reward = -10000
        min_travel_time = float('inf')

        for epoch in range(num_epoch):
            state = np.zeros(self.num_nodes)
            state[start_state] = 1
            battery_charge = np.random.normal(75, 15)
            done = False
            path = [start_state]
            total_reward = 0

            while not done:
                action = self.epsilon_greedy_action(state)
                next_state = np.zeros(self.num_nodes)
                next_state[action] = 1

                reward, battery_charge = self.reward_function(np.argmax(state), action, battery_charge)
                total_reward += reward
                done = (action == end_state or battery_charge <= 0)

                self.store_transition(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                path.append(action)

                if done:
                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_path = path
                    travel_time = self.calculate_travel_time(path)
                    if travel_time < min_travel_time:
                        min_travel_time = travel_time

            self.decay_epsilon()
            self.epoch_rewards.append(total_reward)
            self.travel_times.append(travel_time)

            print(f"Epoch {epoch + 1}/{num_epoch}: Total Reward: {total_reward}, Travel Time: {travel_time}, Epsilon: {self.epsilon}")

        print(f"Best Path: {best_path}, Best Reward: {best_reward}, Minimized Travel Time: {min_travel_time}")
        return best_path, best_reward, min_travel_time

    def decay_epsilon(self):
        """Decay epsilon to reduce exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def calculate_travel_time(self, path):
        """Calculate the travel time based on the path."""
        return sum([self.adjacency_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)])

    def reward_function(self, s_cur, s_next, battery_charge):
        """Reward function similar to Q-learning."""
        battery_consumed = self.adjacency_matrix[int(s_cur)][int(s_next)] * 0.5
        battery_charge -= battery_consumed

        reward = -(2 * self.adjacency_matrix[int(s_cur)][int(s_next)])

        if s_next in self.charging_stations and battery_charge < 20:
            charging_penalty = (80 - battery_charge) * 2
            reward -= charging_penalty
            battery_charge = 80

        if battery_charge < 19:
            reward -= 1000

        return reward, battery_charge

    def plot_results(self):
        """Plot the reward, Q-value convergence, and travel time over the epochs."""
        # Plot Reward per Epoch
        plt.figure()
        plt.plot(self.epoch_rewards)
        plt.xlabel('Epoch')
        plt.ylabel('Total Reward')
        plt.title('Reward per Epoch')
        plt.savefig("reward_per_epoch.png")
        plt.close()

        # Plot Travel Time per Epoch
        plt.figure()
        plt.plot(self.travel_times)
        plt.xlabel('Epoch')
        plt.ylabel('Travel Time')
        plt.title('Travel Time per Epoch')
        plt.savefig("travel_time_per_epoch.png")
        plt.close()


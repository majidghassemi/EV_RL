import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=16, hidden_dim=128):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Added another hidden layer
        self.fc3 = nn.Linear(hidden_dim, num_nodes)

        # Initialize weights using Xavier initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, node_idx, battery_charge):
        x = self.embedding(node_idx)
        battery_charge = battery_charge.unsqueeze(1)
        x = torch.cat([x, battery_charge], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Pass through the added layer
        q_values = self.fc3(x)
        return q_values

class DeepQLearning:
    def __init__(
        self,
        adjacency_matrix,
        num_nodes,
        charging_stations,
        gamma=0.9,
        epsilon=0.2,
        alpha=0.2,
        epsilon_decay_rate=0.999,
        min_epsilon=0.01,
        battery_charge=80,
        replay_buffer_size=50000,  # Increased replay buffer size
        batch_size=128,            # Increased batch size
        target_update_freq=5,      # More frequent target updates
        learning_rate=5e-4,        # Lower learning rate for stable learning
    ):
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.charging_stations = charging_stations
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_battery_charge = battery_charge
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(num_nodes).to(self.device)
        self.target_net = DQN(num_nodes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # L2 regularization

        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size

        self.steps_done = 0
        self.epoch_rewards = []
        self.epoch_distances = []
        self.epoch_battery = []
        self.epoch_travel_times = []
        self.epoch_waiting_times = []
        self.epsilon_values = []
        self.epoch_max_q_values = []
        self.q_convergence = []
        self.epoch_path_lengths = []
        self.epoch_charging_events = []

        if not os.path.exists("epochs"):
            os.makedirs("epochs")

        self.best_epoch_results = {
            "reward": -float("inf"),
            "path": [],
            "battery": 0,
            "distance": 0,
            "travel_time": 0,
            "waiting_time": 0,
        }

    def select_action(self, state, battery_charge):
        current_node = state
        battery_charge_tensor = torch.tensor(
            [battery_charge / self.initial_battery_charge],
            device=self.device,
            dtype=torch.float32,
        )
        valid_actions = np.where(self.adjacency_matrix[int(current_node)] > 0)[0]

        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                current_node_tensor = torch.tensor(
                    [current_node], device=self.device, dtype=torch.long
                )
                q_values = self.policy_net(current_node_tensor, battery_charge_tensor)
                mask = torch.full(
                    (self.num_nodes,), float("-inf"), device=self.device
                )
                mask[valid_actions] = 0.0
                masked_q_values = q_values + mask
                action = masked_q_values.argmax(dim=1).item()

        return action

    def train_agent(self, start_state, end_state, num_epochs, visualize=True, save_video=True):
        print("-" * 20)
        print("Deep Q-Learning begins ...")

        imgs = []
        steps_done = 0

        for epoch in range(1, num_epochs + 1):
            battery_charge = self.initial_battery_charge
            state = start_state
            epoch_reward = 0
            path = [state]
            done = False
            max_q_change = 0

            travel_time = 0
            charging_events = 0
            while not done:
                action = self.select_action(state, battery_charge)

                reward, next_battery_charge = self.reward_function(state, action, battery_charge)
                epoch_reward += reward

                if action == end_state or next_battery_charge <= 0:
                    done = True
                else:
                    done = False

                self.store_transition(state, action, reward, action, done, battery_charge, next_battery_charge)

                q_change = self.train_step()
                if q_change is not None:
                    max_q_change = max(max_q_change, q_change)

                travel_time += self.calculate_travel_time([state, action], base_speed=0.85, traffic_factor=1.33)
                if action in self.charging_stations:
                    charging_events += 1
                state = action
                battery_charge = next_battery_charge
                path.append(state)

                steps_done += 1

                if steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epoch_rewards.append(epoch_reward)
            self.epoch_distances.append(self.cal_distance(path))
            self.epoch_travel_times.append(travel_time)
            self.epoch_waiting_times.append(0)
            self.epoch_battery.append(battery_charge)
            self.epsilon_values.append(self.epsilon)
            self.epoch_path_lengths.append(len(path) - 1)
            self.epoch_charging_events.append(charging_events)
            self.q_convergence.append(max_q_change)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

            print(f"Epoch {epoch}: Total Reward: {epoch_reward}, Distance: {self.epoch_distances[-1]}, "
                  f"Travel Time: {travel_time}, Max Q-Value Change: {max_q_change}, Battery Charge: {battery_charge}, "
                  f"Epsilon: {self.epsilon}")

            if visualize:
                filename = f"epochs/deepq_epoch_{epoch}.png"
                self.plot_graph(
                    src_node=path[0],
                    added_edges=list(zip(path[:-1], path[1:])),
                    figure_title=f"Deep Q-Learning: Epoch {epoch}, Reward: {epoch_reward}",
                    filename=filename,
                )
                imgs.append(filename)

        print(f"Best path for node {start_state} to node {end_state}: {'->'.join(map(str, self.best_epoch_results['path']))}")
        print(f"Best battery charge: {self.best_epoch_results['battery']}")
        print(f"Best reward: {self.best_epoch_results['reward']}")
        print(f"Minimized Travel Time: {self.best_epoch_results['travel_time']}")
        print(f"Total Distance: {self.best_epoch_results['distance']}")

        if visualize and save_video:
            print("Begin to generate gif/mp4 file...")
            images = [imageio.imread(img) for img in imgs]
            imageio.mimsave("deepq-learning.gif", images, fps=5)

        return self.epoch_rewards

    def calculate_travel_time(self, path, base_speed=1.0, traffic_factor=1.0):
        travel_time = 0
        for i in range(len(path) - 1):
            distance = self.adjacency_matrix[path[i]][path[i + 1]]
            speed = base_speed / traffic_factor
            travel_time += distance / speed
        return travel_time

    def cal_distance(self, path):
        dis = 0
        for i in range(len(path) - 1):
            dis += self.adjacency_matrix[path[i]][path[i + 1]]
        return dis

    def reward_function(self, s_cur, s_next, battery_charge):
        battery_consumed = self.adjacency_matrix[int(s_cur)][int(s_next)] * 0.85
        battery_charge -= battery_consumed

        reward = -(2.5 * self.adjacency_matrix[int(s_cur)][int(s_next)])

        if battery_charge < 20:
            reward -= 1000

        if s_next in self.charging_stations and battery_charge < 20:
            charging_penalty = (80 - battery_charge) * 1.5
            reward -= charging_penalty
            battery_charge = 80
        return reward, battery_charge

    def store_transition(self, state, action, reward, next_state, done, battery_charge, next_battery_charge):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done, battery_charge, next_battery_charge))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = list(zip(*batch))

        state_batch = torch.tensor(batch[0], device=self.device, dtype=torch.long)
        action_batch = torch.tensor(batch[1], device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(batch[3], device=self.device, dtype=torch.long)
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.float32)
        battery_charge_batch = (
            torch.tensor(batch[5], device=self.device, dtype=torch.float32)
            / self.initial_battery_charge
        )
        next_battery_charge_batch = (
            torch.tensor(batch[6], device=self.device, dtype=torch.float32)
            / self.initial_battery_charge
        )

        q_values = self.policy_net(state_batch, battery_charge_batch)
        q_values = q_values.gather(1, action_batch).squeeze()

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch, next_battery_charge_batch)
            max_next_q_values = []
            for idx, next_state in enumerate(next_state_batch):
                valid_actions = np.where(
                    self.adjacency_matrix[int(next_state.cpu().numpy())] > 0
                )[0]
                if len(valid_actions) == 0:
                    max_next_q_values.append(torch.tensor(0.0, device=self.device))
                else:
                    mask = torch.full(
                        (self.num_nodes,), float("-inf"), device=self.device
                    )
                    mask[valid_actions] = 0.0
                    masked_q_values = next_q_values[idx] + mask
                    max_next_q_value = masked_q_values.max()
                    max_next_q_values.append(max_next_q_value)
            max_next_q_values = torch.stack(max_next_q_values)

        target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        q_change = torch.abs(q_values - target_q_values).max().item()

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()

        return q_change

    def plot_graph(self, figure_title=None, src_node=None, added_edges=None, filename=None):
        adjacency_matrix = np.array(self.adjacency_matrix)
        rows, cols = np.where(adjacency_matrix > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))
        values = [adjacency_matrix[i][j] for i, j in edges]
        weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]

        plt.cla()
        fig = plt.figure(1)
        if figure_title is None:
            plt.title("The shortest path for every node to the target")
        else:
            plt.title(figure_title)

        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)

        labels = nx.get_edge_attributes(G, "weight")
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_size=15)
        nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
        nodes.set_edgecolor("black")

        if src_node is not None:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[src_node], node_color="g")
        else:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="g")

        nodes.set_edgecolor("black")
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)

        if added_edges is not None:
            nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color="r", width=2)

        if filename:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")

        plt.close(fig)
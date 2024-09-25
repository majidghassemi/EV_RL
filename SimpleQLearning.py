import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import os

class SimpleQLearning:
    def __init__(self, adjacency_matrix, num_nodes, charging_stations, gamma=0.9, epsilon=0.2, alpha=0.1, epsilon_decay_rate=0.98, min_epsilon=0.01, min_alpha=0.001, battery_charge=80):
        self.adjacency_matrix = adjacency_matrix  # Use the adjacency matrix passed in during initialization
        self.num_nodes = num_nodes
        self.charging_stations = charging_stations  # Charging stations list
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_battery_charge = battery_charge  # Renamed for clarity
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.min_alpha = min_alpha
        self.q_convergence = []
        self.epoch_rewards = []
        self.epoch_distances = []
        self.epoch_travel_times = []
        self.epoch_waiting_times = []

        # Initialize Q-table with zeros (no heuristic initialization)
        self.q_table = np.zeros((self.num_nodes, self.num_nodes))

        # Create the epochs directory if it doesn't exist
        if not os.path.exists("epochs"):
            os.makedirs("epochs")

        self.best_epoch_results = {
            "reward": -float('inf'),
            "path": [],
            "battery": 0,
            "distance": 0,
            "travel_time": 0,
            "waiting_time": 0
        }  # To track best epoch
        self.best_travel_time = float('inf')  # To track minimized travel time

    def cal_distance(self, path):
        """Calculate the total distance of a given path."""
        dis = 0
        for i in range(len(path) - 1):
            dis += self.adjacency_matrix[path[i]][path[i + 1]]
        return dis

    def calculate_travel_time(self, path):
        """Calculate total travel time for the given path."""
        travel_time = sum([self.adjacency_matrix[path[i]][path[i+1]] for i in range(len(path) - 1)])
        return travel_time

    def plot_graph(self, figure_title=None, src_node=None, added_edges=None, filename=None):
        """Visualize and save the current graph with the agent's progress."""
        adjacency_matrix = np.array(self.adjacency_matrix)
        rows, cols = np.where(adjacency_matrix > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))
        values = [adjacency_matrix[i][j] for i, j in edges]
        weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]

        plt.cla()  # Clear the current axes
        fig = plt.figure(1)
        if figure_title is None:
            plt.title("The shortest path for every node to the target")
        else:
            plt.title(figure_title)
        
        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)

        labels = nx.get_edge_attributes(G, 'weight')
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_size=15)
        nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
        nodes.set_edgecolor('black')

        # Dynamically set the source node
        if src_node is not None:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[src_node], node_color="g")  # Highlight source node
        else:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="g")  # Fallback to node 0

        nodes.set_edgecolor('black')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)

        if added_edges is not None:
            nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='r', width=2)

        if filename:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")

        plt.close(fig)
    
    def epsilon_greedy(self, s_curr, q):
        """Select next state using epsilon-greedy strategy."""
        potential_next_states = np.where(np.array(self.adjacency_matrix[int(s_curr)]) > 0)[0]
        if len(potential_next_states) == 0:
            return None

        if random.random() > self.epsilon:
            q_of_next_states = [q.get((int(s_curr), int(s_next)), 0) for s_next in potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:
            s_next = random.choice(potential_next_states)

        return int(s_next)

    def epsilon_decay(self, epoch):
        """Decay the epsilon value to reduce exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

    def learning_rate_scheduler(self, epoch, decay_rate=0.99):
        """Decay the learning rate over time to stabilize updates."""
        self.alpha = max(self.min_alpha, self.alpha * decay_rate)

    def reward_function(self, s_cur, s_next, battery_charge):
        """Define the reward function with respect to distance and battery charge."""
        battery_consumed = self.adjacency_matrix[int(s_cur)][int(s_next)] * 0.5
        battery_charge -= battery_consumed

        # Base negative reward for traveling the distance
        reward = -(2 * self.adjacency_matrix[int(s_cur)][int(s_next)])

        # Penalize if battery falls below 20
        if battery_charge < 20:
            reward -= 1000  # Large penalty for battery going below 20

        # If reaching a charging station and battery is below 20, recharge and penalize for waiting time
        if s_next in self.charging_stations and battery_charge < 20:
            charging_penalty = (80 - battery_charge) * 2  # Penalty for the time spent recharging
            reward -= charging_penalty  # Subtract a penalty for recharging
            battery_charge = 80  # Recharge the battery to full capacity (80)
        
        return reward, battery_charge

    def sq_learning(self, start_state, end_state, num_epoch, visualize=True, save_video=True):
        """Run the Q-learning algorithm."""
        print("-" * 20)
        print("sq_learning begins ...")

        best_reward = -10000
        best_battery = 0
        best_path = []
        best_travel_time = float('inf')

        if start_state == end_state:
            raise Exception("start node(state) can't be target node(state)!")

        imgs = []
        q = self.q_table 
        convergence_threshold = 1e-5

        for i in range(1, num_epoch + 1):
            battery_charge = self.initial_battery_charge  # Reset battery_charge at the start of each epoch
            s_cur = start_state
            path = [s_cur]
            max_q_change = 0

            epoch_reward = 0  # Track reward for this epoch
            epoch_distance = 0  # Track distance for this epoch
            epoch_travel_time = 0  # Track travel time for this epoch
            epoch_waiting_time = 0  # Track waiting time for charging

            while True:
                s_next = self.epsilon_greedy(s_cur, q)
                if s_next is None:
                    break

                # Select s_next_next for Q-learning update
                s_next_next = self.epsilon_greedy(s_next, q)
                reward, battery_charge = self.reward_function(s_cur, s_next, battery_charge)
                epoch_reward += reward  # Add to epoch reward

                # Apply Q-learning update
                delta = reward + self.gamma * (q[s_next, s_next_next] if s_next_next is not None else 0) - q[s_cur, s_next]
                q_change = self.alpha * delta
                q[s_cur, s_next] += q_change

                max_q_change = max(max_q_change, abs(q_change))
                s_cur = s_next
                path.append(s_cur)

                # Break the loop if end state is reached or battery is depleted
                if s_cur == end_state or battery_charge <= 0:
                    if best_reward < epoch_reward:
                        best_reward = epoch_reward
                        best_path = path
                        best_battery = battery_charge
                    break

            # Calculate travel time and distance for this epoch
            travel_time = self.calculate_travel_time(path)  # Travel time is now distance * fixed scalar
            distance = self.cal_distance(path)

            # Track epoch data
            self.epoch_distances.append(distance)
            self.epoch_travel_times.append(travel_time)
            self.epoch_waiting_times.append(epoch_waiting_time)

            if travel_time < best_travel_time:
                best_travel_time = travel_time  # Update minimized travel time

            # Track the best epoch results
            if epoch_reward > self.best_epoch_results['reward']:
                self.best_epoch_results = {
                    "reward": epoch_reward,
                    "path": path,
                    "battery": battery_charge,
                    "distance": distance,
                    "travel_time": travel_time,
                    "waiting_time": epoch_waiting_time
                }

            self.q_convergence.append(max_q_change)
            self.epoch_rewards.append(epoch_reward)  # Store total epoch reward
            self.epsilon_decay(i)
            self.learning_rate_scheduler(i)

            print(f"Epoch {i}: Total Reward: {epoch_reward}, Distance: {distance}, Travel Time: {travel_time}, Max Q-Value Change: {max_q_change}, Battery Charge: {battery_charge}, Epsilon: {self.epsilon}")

            if visualize:
                filename = f"epochs/sqlearning_epoch_{i}.png"
                img = self.plot_graph(src_node=start_state,
                                      added_edges=list(zip(path[:-1], path[1:])),
                                      figure_title=f"sq-learning: epoch {i}, reward: {reward}",
                                      filename=filename)
                imgs.append(img)  # Append the image to the list

            if max_q_change < convergence_threshold:
                print(f"Converged after {i} epochs.")
                break

        print(f"Best path for node {start_state} to node {end_state}: {'->'.join(map(str, self.best_epoch_results['path']))}")
        print(f"Best battery charge: {self.best_epoch_results['battery']}")
        print(f"Best reward: {self.best_epoch_results['reward']}")
        print(f"Minimized Travel Time: {self.best_epoch_results['travel_time']}")
        print(f"Total Distance: {self.best_epoch_results['distance']}")

        if visualize and save_video:
            print("Begin to generate gif/mp4 file...")
            imageio.mimsave("sq-learning.gif", imgs, fps=5)

        return self.best_epoch_results


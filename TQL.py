import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pickle
import os

class QLearning:
    def __init__(self, adjacency_matrix, num_nodes, charging_stations, q_values_file=None, gamma=0.9, epsilon=0.2, alpha=0.1, epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.001, battery_charge=80):
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.charging_stations = charging_stations 
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_battery_charge = battery_charge 
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.min_alpha = min_alpha
        self.q_convergence = []
        self.epoch_rewards = []
        self.epoch_distances = []
        self.epoch_travel_times = []
        self.epoch_waiting_times = []

        self.q_table = self.load_q_table(q_values_file) if q_values_file else np.zeros((self.num_nodes, self.num_nodes))

        if not os.path.exists("epochs"):
            os.makedirs("epochs")

        self.best_epoch_results = {
            "reward": -float('inf'),
            "path": [],
            "battery": 0,
            "distance": 0,
            "travel_time": 0,
            "waiting_time": 0
        }
        self.best_travel_time = float('inf')

    def load_q_table(self, q_values_file):
        """Load the Q-table from TERC2 results (pickle file)."""
        with open(q_values_file, 'rb') as f:
            q_table = pickle.load(f)
        
        for s_curr in range(self.num_nodes):
            for s_next in range(self.num_nodes):
                if (s_curr, s_next) not in q_table:
                    q_table[(s_curr, s_next)] = 0
        
        print(f"Q-values loaded from {q_values_file}")
        return q_table

    def cal_distance(self, path):
        dis = 0
        for i in range(len(path) - 1):
            dis += self.adjacency_matrix[path[i]][path[i + 1]]
        return dis

    def calculate_travel_time(self, path, base_speed=0.85, traffic_factor=1.33):
        travel_time = 0
        for i in range(len(path) - 1):
            distance = self.adjacency_matrix[path[i]][path[i + 1]]
            
            # Combine base speed and traffic factor to calculate time for this segment
            speed = base_speed / traffic_factor
            travel_time += distance / speed
        
        return travel_time



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

        labels = nx.get_edge_attributes(G, 'weight')
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_size=15)
        nodes = nx.draw_networkx_nodes(G, pos, node_color="y")
        nodes.set_edgecolor('black')

        if src_node is not None:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[src_node], node_color="g")
        else:
            nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="g")

        nodes.set_edgecolor('black')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)

        if added_edges is not None:
            nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='r', width=2)

        if filename:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img

    def epsilon_greedy(self, s_curr, q):
        potential_next_states = np.where(np.array(self.adjacency_matrix[int(s_curr)]) > 0)[0]
        if len(potential_next_states) == 0:
            return None

        if random.random() > self.epsilon:
            q_of_next_states = [q[int(s_curr), int(s_next)] for s_next in potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:
            s_next = random.choice(potential_next_states)

        return int(s_next)


    def epsilon_decay(self, epoch):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

    def learning_rate_scheduler(self, epoch, decay_rate=0.99):
        self.alpha = max(self.min_alpha, self.alpha * decay_rate)

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


    def q_learning(self, start_state, end_state, num_epoch, visualize=True, save_video=True):
        print("-" * 20)
        print("q_learning begins ...")

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
            battery_charge = self.initial_battery_charge
            s_cur = start_state
            path = [s_cur]
            max_q_change = 0

            epoch_reward = 0
            epoch_distance = 0 
            epoch_travel_time = 0 
            epoch_waiting_time = 0

            while True:
                s_next = self.epsilon_greedy(s_cur, q)
                if s_next is None:
                    break

                s_next_next = self.epsilon_greedy(s_next, q)
                reward, battery_charge = self.reward_function(s_cur, s_next, battery_charge)
                epoch_reward += reward

                if s_next in self.charging_stations and battery_charge < 20:
                    waiting_time = (80 - battery_charge) / 2
                    epoch_waiting_time += waiting_time 
                    battery_charge = 80 

                delta = reward + self.gamma * (q[s_next, s_next_next] if s_next_next is not None else 0) - q[s_cur, s_next]
                q_change = self.alpha * delta
                q[s_cur, s_next] += q_change

                max_q_change = max(max_q_change, abs(q_change))
                s_cur = s_next
                path.append(s_cur)

                if s_cur == end_state or battery_charge <= 0:
                    if best_reward < epoch_reward:
                        best_reward = epoch_reward
                        best_path = path
                        best_battery = battery_charge
                    break

            travel_time = self.calculate_travel_time(path)
            distance = self.cal_distance(path)

            # Track epoch data
            self.epoch_distances.append(distance)
            self.epoch_travel_times.append(travel_time)
            self.epoch_waiting_times.append(epoch_waiting_time)

            if travel_time < best_travel_time:
                best_travel_time = travel_time

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
            self.epoch_rewards.append(epoch_reward)
            self.epsilon_decay(i)
            self.learning_rate_scheduler(i)

            print(f"Epoch {i}: Total Reward: {epoch_reward}, Distance: {distance}, Travel Time: {travel_time}, Waiting Time: {epoch_waiting_time}, Max Q-Value Change: {max_q_change}, Battery Charge: {battery_charge}, Epsilon: {self.epsilon}")

            if visualize:
                filename = f"epochs/qlearning_epoch_{i}.png"
                img = self.plot_graph(src_node=start_state,
                                    added_edges=list(zip(path[:-1], path[1:])),
                                    figure_title=f"q-learning: epoch {i}, reward: {reward}",
                                    filename=filename)
                imgs.append(img)

            if max_q_change < convergence_threshold:
                print(f"Converged after {i} epochs.")
                break

        print(f"Best path for node {start_state} to node {end_state}: {'->'.join(map(str, self.best_epoch_results['path']))}")
        print(f"Best battery charge: {self.best_epoch_results['battery']}")
        print(f"Best reward: {self.best_epoch_results['reward']}")
        print(f"Minimized Travel Time: {self.best_epoch_results['travel_time']}")
        print(f"Total Distance: {self.best_epoch_results['distance']}")
        print(f"Total Waiting Time for Charging: {self.best_epoch_results['waiting_time']}")

        if visualize and save_video:
            print("Begin to generate gif/mp4 file...")
            imageio.mimsave("q-learning.gif", imgs, fps=5)

        return self.best_epoch_results

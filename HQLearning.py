import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pickle
import os

class QLearning:
    def __init__(self, adjacency_matrix, num_nodes, charging_stations, q_values_file=None, gamma=0.9, epsilon=0.05, alpha=0.1, epsilon_decay_rate=0.995, min_epsilon=0.01, min_alpha=0.001):
        self.adjacency_matrix = adjacency_matrix  # Use the adjacency matrix passed in during initialization
        self.num_nodes = num_nodes
        self.charging_stations = charging_stations  # Charging stations list
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.battery_charge = np.random.normal(75, 15)
        # self.battery_charge = max(0, min(100, self.battery_charge))  # Ensure battery charge is between 0 and 100
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.min_alpha = min_alpha
        self.q_convergence = []
        self.epoch_rewards = [] 

        # Initialize Q-table with TERC2 results if provided
        self.q_table = self.load_q_table(q_values_file) if q_values_file else np.zeros((self.num_nodes, self.num_nodes))

        # Create the epochs directory if it doesn't exist
        if not os.path.exists("epochs"):
            os.makedirs("epochs")

    def load_q_table(self, q_values_file):
        """Load the Q-table from TERC2 results (pickle file)."""
        with open(q_values_file, 'rb') as f:
            q_table = pickle.load(f)
        
        # Ensure all state-action pairs are initialized in the table
        for s_curr in range(self.num_nodes):
            for s_next in range(self.num_nodes):
                if (s_curr, s_next) not in q_table:
                    q_table[(s_curr, s_next)] = 0
        
        print(f"Q-values loaded from {q_values_file}")
        return q_table

    def cal_distance(self, path):
        """Calculate the total distance of a given path."""
        dis = 0
        for i in range(len(path) - 1):
            dis += self.adjacency_matrix[path[i]][path[i + 1]]
        return dis

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
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[0, src_node] if src_node else [0], node_color="g")
        nodes.set_edgecolor('black')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=15)

        if added_edges is not None:
            nx.draw_networkx_edges(G, pos, edgelist=added_edges, edge_color='r', width=2)

        if filename:
            plt.savefig(filename)
            print(f"Plot saved as {filename}")

        # Convert figure to numpy array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img  # Return the image array

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
        # Reduce battery by distance traveled
        battery_consumed = self.adjacency_matrix[int(s_cur)][int(s_next)] * 0.5
        battery_charge -= battery_consumed
        
        # Penalize by distance
        reward = -(2 * self.adjacency_matrix[int(s_cur)][int(s_next)])
        
        # Check if the next state is a charging station
        if s_next in self.charging_stations and battery_charge < 20:
            # Add a time-based penalty for charging (optional), and recharge the battery to 80
            charging_penalty = (80 - battery_charge) * 2  # Simulating waiting time at charging station
            reward -= charging_penalty
            battery_charge = 80  # Fully recharge the battery
        
        # Heavy penalty for low battery charge
        if battery_charge <19:
            reward -= 1000  # Heavy penalty for running out of battery
        
        return reward, battery_charge

    def q_learning(self, start_state, end_state, num_epoch, visualize=True, save_video=True):
        """Run the Q-learning algorithm."""
        print("-" * 20)
        print("q_learning begins ...")

        best_reward = -10000
        best_battery = 0
        best_path = []

        if start_state == end_state:
            raise Exception("start node(state) can't be target node(state)!")

        imgs = []
        q = self.q_table 
        convergence_threshold = 1e-5

        for i in range(1, num_epoch + 1):
            reward = 0
            battery_charge = self.battery_charge
            s_cur = start_state
            path = [s_cur]
            len_of_path = 0
            max_q_change = 0

            epoch_reward = 0  # Track reward for this epoch

            while True:
                s_next = self.epsilon_greedy(s_cur, q)
                if s_next is None:
                    break

                s_next_next = self.epsilon_greedy(s_next, q)
                reward, battery_charge = self.reward_function(s_cur, s_next, battery_charge)
                epoch_reward += reward  # Add to epoch reward

                delta = reward + self.gamma * q[s_next, s_next_next] - q[s_cur, s_next]
                q_change = self.alpha * delta
                q[s_cur, s_next] += q_change

                max_q_change = max(max_q_change, abs(q_change))
                s_cur = s_next
                path.append(s_cur)

                print(f"Transition: {s_cur} -> {s_next}, Battery charge: {battery_charge}, Reward: {reward}")

                if s_cur == end_state or battery_charge <= 0:
                    if best_reward < reward:
                        best_reward = reward
                        best_path = path
                        best_battery = battery_charge
                    break

            self.q_convergence.append(max_q_change)
            self.epoch_rewards.append(epoch_reward)  # Store total epoch reward
            self.epsilon_decay(i)
            self.learning_rate_scheduler(i)

            # Logging
            print(f"Epoch {i}: Total Reward: {epoch_reward}, Max Q-Value Change: {max_q_change}, Battery Charge: {battery_charge}, Epsilon: {self.epsilon}")

            if visualize:
                # Save the plot in the "epochs" directory
                filename = f"epochs/qlearning_epoch_{i}.png"
                img = self.plot_graph(src_node=start_state,
                                      added_edges=list(zip(path[:-1], path[1:])),
                                      figure_title=f"q-learning: epoch {i}, reward: {reward}",
                                      filename=filename)
                imgs.append(img)  # Append the image to the list

            if max_q_change < convergence_threshold:
                print(f"Converged after {i} epochs.")
                break

        print(f"Best path for node {start_state} to node {end_state}: {'->'.join(map(str, best_path))}")
        print(f"Battery charge: {best_battery}")
        print(f"Reward: {best_reward}")

        if visualize and save_video:
            print("begin to generate gif/mp4 file...")
            imageio.mimsave("q-learning.gif", imgs, fps=5)

        return best_path, best_reward


# Example of plotting the reward and Q-value convergence:
def plot_learning_metrics(q_learning_instance):
    """Plot the reward per epoch and Q-value convergence."""
    # Plot Q-value convergence
    plt.figure()
    plt.plot(q_learning_instance.q_convergence)
    plt.xlabel('Epoch')
    plt.ylabel('Max Q-Value Change')
    plt.title('Q-Value Convergence')
    plt.savefig('q_value_convergence.png')

    # Plot reward per epoch
    plt.figure()
    plt.plot(q_learning_instance.epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Reward per Epoch')
    plt.savefig('reward_per_epoch.png')

from road_network import RandomGraph
from TERC2 import TERC2
from TQL import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DQN, DeepQLearning
import networkx as nx
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

# Generate the road network using RandomGraph
graph_generator = RandomGraph(100, 0.085, 42, 1)
generated_graph, charging_stations = graph_generator.generate_graph(show_stat=True)

# Initialize TERC2 with the generated graph
terc2 = TERC2(generated_graph)

# Add charging nodes to TERC2
for station in charging_stations:
    terc2.add_charging_nodes(station)

# Export results for Q-learning initialization
terc2.export_terc_results("terc_results.pkl")
terc2.export_to_qlearning("q_values.pkl")

# Extract the adjacency matrix from the generated graph
adjacency_matrix = nx.to_numpy_array(generated_graph)

# Set start and end state
start_state = 67
end_state = 2

# Initialize Q-Learning (TQL)
tql_agent = QLearning(start_state, end_state, adjacency_matrix, "q_values.pkl", alpha=0.1, gamma=0.9, epsilon=0.1)
tql_rewards = []
for i in range(100):
    reward = tql_agent.train()
    tql_rewards.append(reward)

# Initialize Simple Q-Learning
simple_q_agent = SimpleQLearning(start_state, end_state, adjacency_matrix, alpha=0.1, gamma=0.9, epsilon=0.1)
simple_q_rewards = []
for i in range(100):
    reward = simple_q_agent.train()
    simple_q_rewards.append(reward)

# Initialize DQN (Deep Q-Learning)
dqn_agent = DQN(start_state, end_state, adjacency_matrix)
dqn_rewards = []
for i in range(100):
    reward = dqn_agent.train()
    dqn_rewards.append(reward)

# Plot comparison of cumulative rewards for all three algorithms
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(tql_rewards), label="TQL (Q-Learning)")
plt.plot(np.cumsum(simple_q_rewards), label="Simple Q-Learning")
plt.plot(np.cumsum(dqn_rewards), label="DQN (Deep Q-Learning)")
plt.title("Cumulative Reward Comparison")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()

from road_network import RandomGraph
from TERC2 import TERC2
from HQLearning import QLearning
import networkx as nx
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

# Generate the road network using RandomGraph
graph_generator = RandomGraph(45, 0.25, 42, 3)
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
start_state = 93
end_state = 15

# Initialize Q-learning with the adjacency matrix and charging stations
q_learning = QLearning(adjacency_matrix=adjacency_matrix, 
                       num_nodes=len(generated_graph.nodes), 
                       charging_stations=charging_stations,  # Pass charging stations
                       q_values_file="q_values.pkl", 
                       epsilon=0.2,  # Initial epsilon for exploration
                       alpha=0.1,  # Learning rate
                       epsilon_decay_rate=0.995)  # Decay rate for epsilon

# Run Q-learning using start_state and end_state
best_path, best_reward = q_learning.q_learning(start_state=start_state, end_state=end_state, num_epoch=250, visualize=True, save_video=True)

# Output the results
print(f"Best path found by Q-learning: {best_path}")
print(f"Best reward achieved: {best_reward}")

# Plot Q-value convergence and save it
plt.plot(q_learning.q_convergence)
plt.xlabel('Epoch')
plt.ylabel('Max Q-Value Change')
plt.title('Q-Value Convergence')
plt.savefig("q_value_convergence.png")
plt.close()

# Plot reward per epoch and save it
plt.figure()
plt.plot(q_learning.epoch_rewards)
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.title('Reward per Epoch')
plt.savefig("reward_per_epoch.png")
plt.close()

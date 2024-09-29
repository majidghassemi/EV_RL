from road_network import RandomGraph
from TERC2 import TERC2
from HQLearning import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DQNetwork, DQNAgent
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

# HQLEARNING TESTING

# Initialize Q-learning with the adjacency matrix and charging stations
# q_learning = QLearning(adjacency_matrix=adjacency_matrix, 
#                        num_nodes=len(generated_graph.nodes), 
#                        charging_stations=charging_stations,  # Pass charging stations
#                        q_values_file="q_values.pkl")  # Decay rate for epsilon

# # Run Q-learning using start_state and end_state
# best_path, best_reward, best_travel_time, *_ = q_learning.q_learning(start_state=start_state, end_state=end_state, num_epoch=250, visualize=True, save_video=True)

# # Output the results
# print(f"Best path found by TQ Learning: {best_path}")
# print(f"Best reward achieved: {best_reward}")
# print(f"Minimized travel time: {best_travel_time}")

# # Plot Q-value convergence and save it
# plt.plot(q_learning.q_convergence)
# plt.xlabel('Epoch')
# plt.ylabel('Max Q-Value Change')
# plt.title('Q-Value Convergence for TQ learning')
# plt.savefig("q_value_convergence_HQ.png")
# plt.close()

# # Plot reward per epoch and save it
# plt.figure()
# plt.plot(q_learning.epoch_rewards)
# plt.xlabel('Epoch')
# plt.ylabel('Total Reward')
# plt.title('Reward per Epoch for TQ Learning')
# plt.savefig("reward_per_epoch_HQ.png")
# plt.close()

# # Q Learning TESTING

# sq_learning = SimpleQLearning(adjacency_matrix=adjacency_matrix, 
#                        num_nodes=len(generated_graph.nodes), 
#                        charging_stations=charging_stations)  # Decay rate for epsilon

# # Run Simple Q-learning using start_state and end_state
# best_path_q, best_reward_q, best_travel_time_q, *_ = sq_learning.sq_learning(start_state=start_state, end_state=end_state, num_epoch=250, visualize=True, save_video=True)

# # Output the results
# print(f"Best path found by Q Learning: {best_path_q}")
# print(f"Best reward achieved: {best_reward_q}")
# print(f"Minimized travel time: {best_travel_time_q}")

# # Plot Q-value convergence and save it
# plt.plot(sq_learning.q_convergence)
# plt.xlabel('Epoch')
# plt.ylabel('Max Q-Value Change')
# plt.title('Q-Value Convergence for Q learning')
# plt.savefig("q_value_convergence.png")
# plt.close()

# # Plot reward per epoch and save it
# plt.figure()
# plt.plot(sq_learning.epoch_rewards)
# plt.xlabel('Epoch')
# plt.ylabel('Total Reward')
# plt.title('Reward per Epoch for Q learning')
# plt.savefig("reward_per_epoch.png")
# plt.close()

# DQN RESULTS TESTING


# Initialize DQN agent
dqn_agent = DeepQLearning(
    adjacency_matrix=adjacency_matrix, 
    num_nodes=len(generated_graph.nodes), 
    charging_stations=charging_stations
)

# Run DQN using start_state and end_state
best_epoch_results = dqn_agent.train(
    start_state=start_state, 
    end_state=end_state, 
    num_epochs=250  # Note: The parameter is num_epochs, not num_epoch
)

# Extract results
best_path = best_epoch_results['path']
best_reward = best_epoch_results['reward']
min_travel_time = best_epoch_results['travel_time']

# Output the results
print(f"Best path found by DQN: {best_path}")
print(f"Best reward achieved: {best_reward}")
print(f"Minimized travel time: {min_travel_time}")

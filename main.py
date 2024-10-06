import os
from road_network import RandomGraph
from TERC2 import TERC2
from TQL import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DeepQLearning
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Define global variable for the number of epochs
num_epoch = 5

# Create a directory to save plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Generate the road network using RandomGraph
num_nodes = 100
graph_generator = RandomGraph(num_nodes, 0.085, 42, 1)
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
start_state = np.random.randint(0, num_nodes)
end_state = np.random.randint(0, num_nodes)

# Ensure start_state and end_state are not the same
while start_state == end_state:
    end_state = np.random.randint(0, num_nodes)

# ----------------- TQL (Q-Learning) -----------------
tql_agent = QLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations, q_values_file="q_values.pkl", alpha=0.25, gamma=0.9, epsilon=0.25, epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
tql_agent.q_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
tql_rewards = tql_agent.epoch_rewards

# ----------------- Simple Q-Learning -----------------
simple_q_agent = SimpleQLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations, alpha=0.25, gamma=0.9, epsilon=0.25, epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
simple_q_agent.sq_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
simple_q_rewards = simple_q_agent.epoch_rewards

# ----------------- DQN (Deep Q-Learning) -----------------
dqn_agent = DeepQLearning(
    adjacency_matrix=adjacency_matrix,
    num_nodes=num_nodes,
    charging_stations=charging_stations,
    gamma=0.9,              
    epsilon=0.2,            
    alpha=0.001,            
    epsilon_decay_rate=0.999,
    min_epsilon=0.01,       
    battery_charge=80,      
    replay_buffer_size=10000,
    batch_size=64,
    target_update_freq=10,
    learning_rate=1e-3
)

# Collect DQN rewards
dqn_rewards = dqn_agent.train_agent(start_state=start_state, end_state=end_state, num_epochs=num_epoch, visualize=False, save_video=False)

# ----------------- Plotting and Saving Results -----------------

dpi_quality = 700
tql_color = 'black'
ql_color = 'orange'
dqn_color = 'cyan'
# Cumulative Reward Comparison
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(tql_rewards), label="TQL", color=tql_color)
plt.plot(np.cumsum(simple_q_rewards), label="Q-Learning", color=ql_color)
plt.plot(np.cumsum(dqn_rewards), label="DQN", color=dqn_color)
plt.title("Cumulative Reward Comparison")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.savefig('plots/cumulative_reward_comparison.png', dpi=dpi_quality)
plt.close()

# Total Distance Traveled Comparison
plt.figure(figsize=(10, 6))
plt.plot(tql_agent.epoch_distances, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_distances, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_distances, label="DQN", color=dqn_color)
plt.title("Total Distance Traveled Comparison")
plt.xlabel("Episodes")
plt.ylabel("Distance Traveled (KM)")
plt.legend()
plt.savefig('plots/total_distance_comparison.png', dpi=dpi_quality)
plt.close()

# Travel Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(tql_agent.epoch_travel_times, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_travel_times, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_travel_times, label="DQN", color=dqn_color)
plt.title("Travel Time Comparison")
plt.xlabel("Episodes")
plt.ylabel("Travel Time (Minutes)")
plt.legend()
plt.savefig('plots/travel_time_comparison.png', dpi=dpi_quality)
plt.close()

# Waiting Time at Charging Stations Comparison
plt.figure(figsize=(10, 6))
plt.plot(tql_agent.epoch_waiting_times, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_waiting_times, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_waiting_times, label="DQN", color=dqn_color)
plt.title("Waiting Time at Charging Stations Comparison")
plt.xlabel("Episodes")
plt.ylabel("Waiting Time (at Charging Stations)")
plt.legend()
plt.savefig('plots/waiting_time_comparison.png', dpi=dpi_quality)
plt.close()

# Reward Per Episode Comparison
plt.figure(figsize=(10, 6))
plt.plot(tql_rewards, label="TQL", color=tql_color)
plt.plot(simple_q_rewards, label="Q-Learning", color=ql_color)
plt.plot(dqn_rewards, label="DQN", color=dqn_color)
plt.title("Reward Per Episode Comparison")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.savefig('plots/reward_per_episode_comparison.png', dpi=dpi_quality)
plt.close()

print("All plots have been saved in the 'plots' directory.")

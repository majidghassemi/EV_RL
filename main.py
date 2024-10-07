import os
from road_network import RandomGraph
from TERC2 import TERC2
from TQL import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DeepQLearning
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

num_epoch = 5

if not os.path.exists("plots"):
    os.makedirs("plots")

num_nodes = 125
graph_generator = RandomGraph(num_nodes, 0.065, 42, 1)
generated_graph, charging_stations = graph_generator.generate_graph(show_stat=True)

terc2 = TERC2(generated_graph)

for station in charging_stations:
    terc2.add_charging_nodes(station)

terc2.export_terc_results("terc_results.pkl")
terc2.export_to_qlearning("q_values.pkl")

adjacency_matrix = nx.to_numpy_array(generated_graph)

start_state = np.random.randint(0, num_nodes)
end_state = np.random.randint(0, num_nodes)

while start_state == end_state:
    end_state = np.random.randint(0, num_nodes)

tql_agent = QLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations, q_values_file="q_values.pkl", alpha=0.25, gamma=0.9, epsilon=0.25, epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
tql_agent.q_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
tql_rewards = tql_agent.epoch_rewards

simple_q_agent = SimpleQLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations, alpha=0.25, gamma=0.9, epsilon=0.25, epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
simple_q_agent.sq_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
simple_q_rewards = simple_q_agent.epoch_rewards

dqn_agent = DeepQLearning(
    adjacency_matrix=adjacency_matrix,
    num_nodes=num_nodes,
    charging_stations=charging_stations,
    gamma=0.9,              
    epsilon=0.2,            
    alpha=0.01,            
    epsilon_decay_rate=0.999,
    min_epsilon=0.01,       
    battery_charge=80,      
    replay_buffer_size=10000,
    batch_size=64,
    target_update_freq=10,
    learning_rate=1e-3
)

dqn_rewards = dqn_agent.train_agent(start_state=start_state, end_state=end_state, num_epochs=num_epoch, visualize=False, save_video=False)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

tql_normalized_rewards = normalize(tql_rewards)
simple_q_normalized_rewards = normalize(simple_q_rewards)
dqn_normalized_rewards = normalize(dqn_rewards)

tql_normalized_distances = normalize(tql_agent.epoch_distances)
simple_q_normalized_distances = normalize(simple_q_agent.epoch_distances)
dqn_normalized_distances = normalize(dqn_agent.epoch_distances)

tql_normalized_travel_times = normalize(tql_agent.epoch_travel_times)
simple_q_normalized_travel_times = normalize(simple_q_agent.epoch_travel_times)
dqn_normalized_travel_times = normalize(dqn_agent.epoch_travel_times)

dpi_quality = 700
tql_color = 'black'
ql_color = 'orange'
dqn_color = 'cyan'
font_size_labels = 17
font_size_ticks = 19
font_size_legend = 14
fig_size_width = 8
fig_size_height = 12

plot_dir = f'plots/{num_nodes}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_normalized_rewards, label="TQL", color=tql_color)
plt.plot(simple_q_normalized_rewards, label="Q-Learning", color=ql_color)
plt.plot(dqn_normalized_rewards, label="DQN", color=dqn_color)
plt.title(f"Normalized Reward Comparison ({num_nodes})", fontsize=font_size_labels)
plt.xlabel("Episodes", fontsize=font_size_labels)
plt.ylabel("Normalized Reward", fontsize=font_size_labels)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.legend(fontsize=font_size_legend)
plt.savefig(f'{plot_dir}/normalized_reward_comparison.png', dpi=dpi_quality)
plt.close()

plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_normalized_distances, label="TQL", color=tql_color)
plt.plot(simple_q_normalized_distances, label="Q-Learning", color=ql_color)
plt.plot(dqn_normalized_distances, label="DQN", color=dqn_color)
plt.title(f"Normalized Distance Traveled Comparison ({num_nodes})", fontsize=font_size_labels)
plt.xlabel("Episodes", fontsize=font_size_labels)
plt.ylabel("Normalized Distance Traveled", fontsize=font_size_labels)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.legend(fontsize=font_size_legend)
plt.savefig(f'{plot_dir}/normalized_distance_comparison.png', dpi=dpi_quality)
plt.close()

plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_normalized_travel_times, label="TQL", color=tql_color)
plt.plot(simple_q_normalized_travel_times, label="Q-Learning", color=ql_color)
plt.plot(dqn_normalized_travel_times, label="DQN", color=dqn_color)
plt.title(f"Normalized Travel Time Comparison ({num_nodes})", fontsize=font_size_labels)
plt.xlabel("Episodes", fontsize=font_size_labels)
plt.ylabel("Normalized Travel Time", fontsize=font_size_labels)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.legend(fontsize=font_size_legend)
plt.savefig(f'{plot_dir}/normalized_travel_time_comparison.png', dpi=dpi_quality)
plt.close()

plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_agent.q_convergence, label="TQL Max Q-Value Change", color=tql_color)
plt.plot(simple_q_agent.q_convergence, label="Q-Learning Max Q-Value Change", color=ql_color)
plt.plot(dqn_agent.q_convergence, label="DQN Max Q-Value Change", color=dqn_color)
plt.title(f"Max Q-Value Change Over Time ({num_nodes})", fontsize=font_size_labels)
plt.xlabel("Episodes", fontsize=font_size_labels)
plt.ylabel("Max Q-Value Change", fontsize=font_size_labels)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.legend(fontsize=font_size_legend)
plt.savefig(f'{plot_dir}/q_value_change_comparison.png', dpi=dpi_quality)
plt.close()

print("All normalized plots have been saved in the 'plots' directory.")

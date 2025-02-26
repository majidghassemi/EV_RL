import os
from road_network import RandomGraph
from TERC2 import TERC2
from TQL import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DeepQLearning
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Updated font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 22,                     # Base font size
    'axes.titlesize': 36,                # Title font size
    'axes.labelsize': 36,                # Axis labels font size
    'legend.fontsize': 22,               # Legend font size
    'axes.labelpad': 10,                 # Extra space for axis labels
})

num_epoch = 2000

if not os.path.exists("plots"):
    os.makedirs("plots")

num_nodes = 500
graph_generator = RandomGraph(num_nodes, 0.09, 36, 1)
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

tql_agent = QLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations,
                      q_values_file="q_values.pkl", alpha=0.1, gamma=0.9, epsilon=0.25,
                      epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
tql_agent.q_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
tql_rewards = tql_agent.epoch_rewards

simple_q_agent = SimpleQLearning(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, charging_stations=charging_stations,
                                 alpha=0.1, gamma=0.9, epsilon=0.25,
                                 epsilon_decay_rate=0.999, min_epsilon=0.01, min_alpha=0.01)
simple_q_agent.sq_learning(start_state=start_state, end_state=end_state, num_epoch=num_epoch, visualize=False, save_video=False)
simple_q_rewards = simple_q_agent.epoch_rewards

dqn_agent = DeepQLearning(
    adjacency_matrix=adjacency_matrix,
    num_nodes=num_nodes,
    charging_stations=charging_stations,
    gamma=0.9,              
    epsilon=0.2,            
    alpha=0.05,            
    epsilon_decay_rate=0.999,
    min_epsilon=0.01,       
    battery_charge=80,      
    replay_buffer_size=10000,
    batch_size=64,
    target_update_freq=10,
    learning_rate=1e-3
)

dqn_rewards = dqn_agent.train_agent(start_state=start_state, end_state=end_state, num_epochs=num_epoch,
                                    visualize=False, save_video=False)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

tql_normalized_q_convergence = normalize(tql_agent.q_convergence)
simple_q_normalized_q_convergence = normalize(simple_q_agent.q_convergence)
dqn_normalized_q_convergence = normalize(dqn_agent.q_convergence)

dpi_quality = 500
tql_color = '#377eb8'  # Blue (Colorblind-friendly)
ql_color  = '#e41a1c'  # Red (Improved contrast)
dqn_color = '#4daf4a'  # Green (More vibrant)
font_size_labels = 24
font_size_ticks = 40
font_size_legend = 22
fig_size_width = 8
fig_size_height = 12

plot_dir = f'plots/{num_nodes}/diff_chargers/60'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Function to apply improved axis settings
def format_axes():
    plt.gca().tick_params(axis='both', which='both', labelsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)  # Larger x-axis numbers
    plt.yticks(fontsize=font_size_ticks)  # Larger y-axis numbers

# Reward Comparison Plot
plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_rewards, label="TQL", color=tql_color)
plt.plot(simple_q_rewards, label="Q-Learning", color=ql_color)
plt.plot(dqn_rewards, label="DQL", color=dqn_color)
# plt.title(f"Cumulaative Reward ({num_nodes} nodes)")
plt.xlabel("Episodes", labelpad=10)
plt.ylabel("Reward", labelpad=10)
format_axes()
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/reward_comparison.png', dpi=dpi_quality)
plt.close()

# Distance Comparison Plot
plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_agent.epoch_distances, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_distances, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_distances, label="DQL", color=dqn_color)
# plt.title(f"Total Distance Traveled ({num_nodes} nodes)")
plt.xlabel("Episodes", labelpad=10)
plt.ylabel("Distance Traveled (KM)", labelpad=10)
format_axes()
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/distance_comparison.png', dpi=dpi_quality)
plt.close()

# Travel Time Comparison Plot
plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_agent.epoch_travel_times, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_travel_times, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_travel_times, label="DQL", color=dqn_color)
# plt.title(f"Travel Time ({num_nodes} nodes)")
plt.xlabel("Episodes", labelpad=10)
plt.ylabel("Travel Time (Minutes)", labelpad=10)
format_axes()
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/travel_time_comparison.png', dpi=dpi_quality)
plt.close()

# Waiting Time Comparison Plot
plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_agent.epoch_waiting_times, label="TQL", color=tql_color)
plt.plot(simple_q_agent.epoch_waiting_times, label="Q-Learning", color=ql_color)
plt.plot(dqn_agent.epoch_waiting_times, label="DQL", color=dqn_color)
# plt.title(f"Waiting Time at Charging Stations Comparison ({num_nodes} nodes)")
plt.xlabel("Episodes", labelpad=10)
plt.ylabel("Waiting Time (at Charging Stations)", labelpad=10)
format_axes()
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/waiting_time_comparison.png', dpi=dpi_quality)
plt.close()

# Q-Value Change Comparison Plot
plt.figure(figsize=(fig_size_height, fig_size_width))
plt.plot(tql_normalized_q_convergence, label="TQL", color=tql_color)
plt.plot(simple_q_normalized_q_convergence, label="Q-Learning", color=ql_color)
plt.plot(dqn_normalized_q_convergence, label="DQL", color=dqn_color)
# plt.title(f"Normalized Max Q-Value Change Over Time ({num_nodes} nodes)")
plt.xlabel("Episodes", labelpad=10)
plt.ylabel("Q-Value Change", labelpad=10)
format_axes()
plt.legend()
plt.tight_layout()
plt.savefig(f'{plot_dir}/q_value_change_comparison.png', dpi=dpi_quality)
plt.close()

print("All plots have been saved in the 'plots' directory.")

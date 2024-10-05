# from road_network import RandomGraph
# from TERC2 import TERC2
# from HQLearning import QLearning
# from SimpleQLearning import SimpleQLearning
# # from DQN import DQNetwork, DQNAgent
# from DQN import DQN, DeepQLearning
# import networkx as nx
# import pickle
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# # Generate the road network using RandomGraph
# graph_generator = RandomGraph(100, 0.085, 42, 1)
# generated_graph, charging_stations = graph_generator.generate_graph(show_stat=True)

# # Initialize TERC2 with the generated graph
# terc2 = TERC2(generated_graph)

# # Add charging nodes to TERC2
# for station in charging_stations:
#     terc2.add_charging_nodes(station)

# # Export results for Q-learning initialization
# terc2.export_terc_results("terc_results.pkl")
# terc2.export_to_qlearning("q_values.pkl")

# # Extract the adjacency matrix from the generated graph
# adjacency_matrix = nx.to_numpy_array(generated_graph)

# # Set start and end state
# start_state = 67
# end_state = 2

# # HQLEARNING TESTING

# # Initialize Q-learning with the adjacency matrix and charging stations
# # q_learning = QLearning(adjacency_matrix=adjacency_matrix, 
# #                        num_nodes=len(generated_graph.nodes), 
# #                        charging_stations=charging_stations,  # Pass charging stations
# #                        q_values_file="q_values.pkl")  # Decay rate for epsilon

# # # Run Q-learning using start_state and end_state
# # best_path, best_reward, best_travel_time, *_ = q_learning.q_learning(start_state=start_state, end_state=end_state, num_epoch=250, visualize=True, save_video=True)

# # # Output the results
# # print(f"Best path found by TQ Learning: {best_path}")
# # print(f"Best reward achieved: {best_reward}")
# # print(f"Minimized travel time: {best_travel_time}")

# # # Plot Q-value convergence and save it
# # plt.plot(q_learning.q_convergence)
# # plt.xlabel('Epoch')
# # plt.ylabel('Max Q-Value Change')
# # plt.title('Q-Value Convergence for TQ learning')
# # plt.savefig("q_value_convergence_HQ.png")
# # plt.close()

# # # Plot reward per epoch and save it
# # plt.figure()
# # plt.plot(q_learning.epoch_rewards)
# # plt.xlabel('Epoch')
# # plt.ylabel('Total Reward')
# # plt.title('Reward per Epoch for TQ Learning')
# # plt.savefig("reward_per_epoch_HQ.png")
# # plt.close()

# # # Q Learning TESTING

# # sq_learning = SimpleQLearning(adjacency_matrix=adjacency_matrix, 
# #                        num_nodes=len(generated_graph.nodes), 
# #                        charging_stations=charging_stations)  # Decay rate for epsilon

# # # Run Simple Q-learning using start_state and end_state
# # best_path_q, best_reward_q, best_travel_time_q, *_ = sq_learning.sq_learning(start_state=start_state, end_state=end_state, num_epoch=250, visualize=True, save_video=True)

# # # Output the results
# # print(f"Best path found by Q Learning: {best_path_q}")
# # print(f"Best reward achieved: {best_reward_q}")
# # print(f"Minimized travel time: {best_travel_time_q}")

# # # Plot Q-value convergence and save it
# # plt.plot(sq_learning.q_convergence)
# # plt.xlabel('Epoch')
# # plt.ylabel('Max Q-Value Change')
# # plt.title('Q-Value Convergence for Q learning')
# # plt.savefig("q_value_convergence.png")
# # plt.close()

# # # Plot reward per epoch and save it
# # plt.figure()
# # plt.plot(sq_learning.epoch_rewards)
# # plt.xlabel('Epoch')
# # plt.ylabel('Total Reward')
# # plt.title('Reward per Epoch for Q learning')
# # plt.savefig("reward_per_epoch.png")
# # plt.close()

# # DQN RESULTS TESTING


# # Initialize DQN agent
# dqn_agent = DeepQLearning(
#     adjacency_matrix=adjacency_matrix, 
#     num_nodes=len(generated_graph.nodes), 
#     charging_stations=charging_stations
# )

# # # Run DQN using start_state and end_state
# # best_epoch_results = dqn_agent.train(
# #     start_state=start_state, 
# #     end_state=end_state, 
# #     num_epochs=250  # Note: The parameter is num_epochs, not num_epoch
# # )

# best_path, best_reward, min_travel_time = dqn_agent.dqn_learning(start_state=start_state, end_state=end_state, num_epoch=250)


# # Extract results
# best_path = best_epoch_results['path']
# best_reward = best_epoch_results['reward']
# min_travel_time = best_epoch_results['travel_time']


# # Output the results
# print(f"Best path found by DQN: {best_path}")
# print(f"Best reward achieved: {best_reward}")
# print(f"Minimized travel time: {min_travel_time}")





# Import the relevant classes and libraries
from road_network import RandomGraph
from TERC2 import TERC2
from HQLearning import QLearning
from SimpleQLearning import SimpleQLearning
from DQN import DQN, DeepQLearning
import networkx as nx
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

# Define node sizes
node_sizes = [125, 250, 500]

# Initialize results storage
results = {
    'q_learning': {},
    'simple_q_learning': {},
    'dqn': {}
}

# Loop over each node size
for nodes in node_sizes:
    print(f"Running experiment for {nodes} nodes...")

    # Generate the road network for the current node size
    graph_generator = RandomGraph(nodes, 0.085, 42, 1)
    generated_graph, charging_stations = graph_generator.generate_graph(show_stat=False)

    # Export results for Q-learning initialization
    terc2 = TERC2(generated_graph)
    for station in charging_stations:
        terc2.add_charging_nodes(station)
    terc2.export_terc_results("terc_results.pkl")
    terc2.export_to_qlearning("q_values.pkl")

    # Extract the adjacency matrix from the generated graph
    adjacency_matrix = nx.to_numpy_array(generated_graph)

    # Set random start and end states for each run
    start_state = random.randint(0, nodes - 1)
    end_state = random.randint(0, nodes - 1)

    # Run Q-Learning
    q_learning_agent = QLearning(adjacency_matrix=adjacency_matrix,
                                 num_nodes=len(generated_graph.nodes),
                                 charging_stations=charging_stations)
    q_learning_results = q_learning_agent.q_learning(start_state=start_state,
                                                     end_state=end_state,
                                                     num_epoch=250)
    results['q_learning'][nodes] = q_learning_results

    # Run Simple Q-Learning
    simple_q_learning_agent = SimpleQLearning(adjacency_matrix=adjacency_matrix,
                                              num_nodes=len(generated_graph.nodes),
                                              charging_stations=charging_stations)
    simple_q_results = simple_q_learning_agent.sq_learning(start_state=start_state,
                                                           end_state=end_state,
                                                           num_epoch=250)
    results['simple_q_learning'][nodes] = simple_q_results

    # Run DQN
    dqn_agent = DeepQLearning(adjacency_matrix=adjacency_matrix,
                              num_nodes=len(generated_graph.nodes),
                              charging_stations=charging_stations)
    dqn_results = dqn_agent.train(start_state=start_state,
                                  end_state=end_state,
                                  num_epochs=250)
    results['dqn'][nodes] = dqn_results

# Save the results for later use
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Visualization Section

# Load the stored results
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

methods = ['q_learning', 'simple_q_learning', 'dqn']
node_sizes = [125, 250, 500]

# --- 1. Plot Travel Time per Epoch ---
plt.figure(figsize=(8, 6))
for method in methods:
    for nodes in node_sizes:
        travel_times = results[method][nodes]['epoch_travel_times']
        plt.plot(range(len(travel_times)), travel_times, label=f'{method} - {nodes} nodes')

plt.xlabel('Episode')
plt.ylabel('Travel Time')
plt.title('Travel Time per Epoch for Different Node Sizes')
plt.legend()
plt.savefig('travel_time_per_epoch.png')
plt.show()

# --- 2. Final Reward Comparison Across Node Sizes ---
final_rewards = {method: [results[method][nodes]['reward'] for nodes in node_sizes] for method in methods}

fig, ax = plt.subplots(figsize=(8, 6))
width = 0.25  # Width of the bars
x = np.arange(len(node_sizes))

for idx, method in enumerate(methods):
    ax.bar(x + idx * width, final_rewards[method], width, label=method)

ax.set_xlabel('Node Size')
ax.set_ylabel('Total Reward')
ax.set_title('Final Reward Comparison Across Node Sizes')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(node_sizes)
ax.legend()
plt.savefig('final_reward_comparison.png')
plt.show()

# --- 3. Max Q-Value Change per Epoch for Different Node Sizes ---
plt.figure(figsize=(8, 6))
for method in methods:
    for nodes in node_sizes:
        q_value_changes = results[method][nodes]['q_value_changes']
        plt.plot(range(len(q_value_changes)), q_value_changes, label=f'{method} - {nodes} nodes')

plt.xlabel('Episode')
plt.ylabel('Max Q-Value Change')
plt.title('Max Q-Value Change per Epoch for Different Node Sizes')
plt.legend()
plt.savefig('max_q_value_change_per_epoch.png')
plt.show()

# --- 4. Distance Traveled per Epoch for Different Node Sizes ---
plt.figure(figsize=(8, 6))
for method in methods:
    for nodes in node_sizes:
        distances = results[method][nodes]['epoch_distances']
        plt.plot(range(len(distances)), distances, label=f'{method} - {nodes} nodes')

plt.xlabel('Episode')
plt.ylabel('Distance Traveled')
plt.title('Distance Traveled per Epoch for Different Node Sizes')
plt.legend()
plt.savefig('distance_traveled_per_epoch.png')
plt.show()


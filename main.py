import networkx as nx
from road_network import RandomGraph
from TERC2 import TERC2
from HQLearning import QLearning

# Generate the road network using RandomGraph
graph_generator = RandomGraph(33, 0.25, 42, 3)
generated_graph, charging_stations = graph_generator.generate_graph(show_stat=True)

# Initialize TERC2 with the generated graph
terc2 = TERC2(generated_graph)

# Add charging nodes to TERC2
for station in charging_stations:
    terc2.add_charging_nodes(station)

# Export results for Q-learning initialization
terc2.export_to_qlearning("q_values.pkl")
terc2.export_terc_results("terc_results.pkl")

# Extract the adjacency matrix from the generated graph (use to_numpy_array from networkx)
adjacency_matrix = nx.to_numpy_array(generated_graph)

# Initialize Q-learning with precomputed Q-values from TERC2
q_learning = QLearning(adjacency_matrix=adjacency_matrix, 
                       num_nodes=len(generated_graph.nodes), 
                       q_values_file="q_values.pkl")

# Run Q-learning
best_path, best_reward = q_learning.q_learning(start_state=3, end_state=0, num_epoch=750, visualize=True, save_video=False)

# Outputting the results
print(f"Best path found by Q-learning: {best_path}")
print(f"Best reward achieved: {best_reward}")

# Plot Q-value convergence and save it
import matplotlib.pyplot as plt
plt.plot(q_learning.q_convergence)
plt.xlabel('Epoch')
plt.ylabel('Max Q-Value Change')
plt.title('Q-Value Convergence')
plt.savefig("q_value_convergence.png")
plt.close()

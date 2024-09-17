import networkx as nx
from road_network import RandomGraph
from TERC2 import TERC2
from HQLearning import QLearning

graph_generator = RandomGraph(33, 0.25, 42, 3)
generated_graph, charging_stations = graph_generator.generate_graph(show_stat=False)

terc2 = TERC2(generated_graph)

for station in charging_stations:
    terc2.add_charging_nodes(station)

terc2.export_to_qlearning("q_values.pkl")
terc2.export_terc_results("terc_results.pkl")


q_learning = QLearning(adjacency_matrix=nx.to_numpy_array(generated_graph), 
                       num_nodes=len(generated_graph.nodes), 
                       q_values_file="q_values.pkl")

best_path, best_reward = q_learning.q_learning(start_state=0, end_state=0, num_epoch=750, visualize=True, save_video=False)

# Outputting the results
print(f"Best path found by Q-learning: {best_path}")
print(f"Best reward achieved: {best_reward}")
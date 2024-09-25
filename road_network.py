import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle

class RandomGraph:
    def __init__(self, total_nodes, probability, seed, number_of_graphs=1):
        self.total_nodes = total_nodes
        self.charging_station = {}
        self.non_charging_stations = {}
        self.G = {}
        self.number_of_graphs = number_of_graphs
        random.seed(seed)
        np.random.seed(seed)

        # Generate initial graphs
        for i in range(0, self.number_of_graphs):
            self.G[i] = nx.erdos_renyi_graph(total_nodes, probability, seed=seed, directed=False)
            while not nx.is_connected(self.G[i]):
                self.G[i] = nx.erdos_renyi_graph(total_nodes, probability, seed=seed, directed=False)
            first = i * total_nodes
            last = (i + 1) * total_nodes
            mapping = dict(zip(self.G[i], range(first, last)))
            self.G[i] = nx.relabel_nodes(self.G[i], mapping)

    def generate_graph(self, show_stat=False):
        for q in range(0, self.number_of_graphs):
            for (u, v, w) in self.G[q].edges(data=True):
                # Assign distance based on normal distribution (mean 100, std deviation 15)
                ran = max(1, int(np.random.normal(100, 15)))
                w['distance'] = ran
                w['battery_reduction'] = int(0.3 * ran)

            self.charging_station[q] = []
            checked_nodes = []
            candidate_nodes = []
            unchecked_nodes = []
            latest_station = q * self.total_nodes

            checked_nodes.append(latest_station)
            self.charging_station[q].append(latest_station)

            while len(checked_nodes) < self.total_nodes:
                max_distance = 0
                G2 = nx.generators.ego_graph(self.G[q], latest_station, radius=45, distance="battery_reduction")
                for a in list(G2.nodes):
                    if a not in checked_nodes:
                        distance = nx.shortest_path_length(self.G[q], source=a, target=latest_station, weight="battery_reduction", method='dijkstra')
                        if distance > max_distance:
                            candidate_nodes.append(a)
                            max_distance = distance
                        checked_nodes.append(a)
                        checked_nodes = list(dict.fromkeys(checked_nodes))
                if len(candidate_nodes) >= 1:
                    latest_station = candidate_nodes[-1]
                    self.charging_station[q].append(latest_station)
                    checked_nodes.append(latest_station)
                    checked_nodes = list(dict.fromkeys(checked_nodes))
                    candidate_nodes = []
                else:
                    unchecked_nodes = list(set(self.G[q].nodes) - set(checked_nodes))
                    latest_station = random.choice(unchecked_nodes)
                    self.charging_station[q].append(latest_station)
                    checked_nodes.append(latest_station)
                    checked_nodes = list(dict.fromkeys(checked_nodes))
                    candidate_nodes = []

            self.charging_station[q].extend(unchecked_nodes)

        new_G = nx.Graph()
        new_charging_station = []

        for q in range(0, self.number_of_graphs):
            i = self.total_nodes * q
            new_charging_station.extend(self.charging_station[q])
            while i < self.total_nodes * (q + 1):
                node_neighbors = [n for n in self.G[q].neighbors(i)]
                for node in node_neighbors:
                    e = (i, node)
                    edge_data = self.G[q].get_edge_data(i, node)
                    new_G.add_edge(i, node, distance=edge_data.get("distance"), battery_reduction=edge_data.get("battery_reduction"))
                i += 1

        if self.number_of_graphs > 1:
            self.connect_multiple_graphs(new_G)

        if show_stat:
            self.show_graph_statistics(new_G, new_charging_station)

        self.visualize_graph(new_G, new_charging_station)

        return new_G, new_charging_station


    def connect_multiple_graphs(self, new_G):
        for i in range(self.number_of_graphs - 1):
            num_connections = min(len(self.charging_station[i]), len(self.charging_station[i+1]))
            for j in range(num_connections):
                ran = random.randint(50, 150)
                new_G.add_edge(self.charging_station[i][j], self.charging_station[i+1][j],
                               distance=ran, battery_reduction=int(0.3 * ran))
                print(f"Connected charging station {self.charging_station[i][j]} in graph {i} "
                      f"with charging station {self.charging_station[i+1][j]} in graph {i+1} "
                      f"with distance {ran} and battery reduction {int(0.3 * ran)}")

    def show_graph_statistics(self, new_G, new_charging_station):
        new_non_charging_stations = [a for a in list(new_G.nodes) if a not in new_charging_station]
        print(f"Count of non-charging stations: {len(new_non_charging_stations)}")
        print(f"Count of charging stations: {len(new_charging_station)}")
        print("Set of charging stations:", new_charging_station)
        print("Set of non-charging stations:", new_non_charging_stations)

        for i in range(self.total_nodes * self.number_of_graphs):
            node_neighbors = [n for n in new_G.neighbors(i)]
            edges = ", ".join([f"<{i},{n}>" for n in node_neighbors])
            print(f"Edges for node {i}: {edges}")

    def visualize_graph(self, new_G, new_charging_station):
        val_map = {0: 1.0}
        for node in new_charging_station:
            val_map[node] = 1.0
        values = [val_map.get(node, 0.25) for node in new_G.nodes()]
        nx.draw(new_G, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white', linewidths=15, node_size=100)
        plt.show()

    def export_graph(self, graph, filename):
        """Exports the graph to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Graph saved to {filename}")

    def import_graph(self, filename):
        """Imports a graph from a file using pickle."""
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        print(f"Graph loaded from {filename}")
        return graph


# Example usage
graph_generator = RandomGraph(100, 0.25, 42, 1)
generated_graph, charging_stations = graph_generator.generate_graph(show_stat=True)

# Export the generated graph
graph_generator.export_graph(generated_graph, "road_network.gpickle")

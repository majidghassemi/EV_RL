import networkx as nx
import pickle
import matplotlib as plt
import time
import random

class TERC2:
    def __init__(self, nxGraph):
        self.graph = nxGraph
        self.charging_nodes = []

    def add_charging_nodes(self, node):
        self.charging_nodes.append(node)

    def add_edge(self, from_node, to_node, weight, battery_reduction):
        self.graph.add_edge(from_node, to_node, weight=weight, battery_reduction=battery_reduction)

    def draw_graph(self):
        pos = nx.spring_layout(self.graph, seed=7)
        nx.draw_networkx_nodes(self.graph, pos, node_size=1000)
        nx.draw_networkx_edges(self.graph, pos, width=6)
        nx.draw_networkx_labels(self.graph, pos, font_size=20, font_family="sans-serif")
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def shortest_path(self, source, destination):
        path = nx.dijkstra_path(self.graph, source, destination, weight="distance")
        distance = nx.dijkstra_path_length(self.graph, source, destination, weight="distance")
        return path, distance

    def choose_station(self, source, destination):
        best_station = None
        lowest_total = float('inf')
        all_available_charging_stations = [n for n in self.charging_nodes if n != source]

        for station in all_available_charging_stations:
            t1 = nx.shortest_path_length(self.graph, source, station, weight="distance")
            t2 = nx.shortest_path_length(self.graph, station, destination, weight="distance")
            total = t1 + t2

            if total < lowest_total:
                if nx.shortest_path_length(self.graph, source, station, weight="battery_reduction") < 80:
                    lowest_total = total
                    best_station = station

        return best_station

    def total_battery_reduction(self, source, destination):
        total_battery = 0
        path = nx.dijkstra_path(self.graph, source, destination, weight="distance")
        for i in range(len(path) - 1):
            e = (path[i], path[i+1])
            edge_data = self.graph.get_edge_data(*e)
            total_battery += edge_data.get("battery_reduction", 0)
        return total_battery

    def complete_logic(self, source, destination, battery_level):
        start_time = time.time()
        spent_time = 0
        path = []
        battery_charge = battery_level - self.total_battery_reduction(source, destination)

        if battery_charge >= 25:
            spent_time = nx.dijkstra_path_length(self.graph, source, destination, weight="distance")
            path = nx.dijkstra_path(self.graph, source, destination, weight="distance")
            return spent_time, self.total_battery_reduction(source, destination), path, battery_charge

        last_source = source
        total_charge_percent = 0
        total_distance = 0
        total_battery_consumption = 0
        while True:
            new_station = self.choose_station(last_source, destination)
            spent_time += nx.dijkstra_path_length(self.graph, last_source, new_station, weight="distance")
            total_charge_percent += self.total_battery_reduction(last_source, new_station)

            sub_path = nx.dijkstra_path(self.graph, last_source, new_station, weight="distance")
            path.append(sub_path)
            total_distance += nx.path_weight(self.graph, sub_path, "distance")
            total_battery_consumption += nx.path_weight(self.graph, sub_path, "battery_reduction")
            last_source = new_station
            battery_level = 100

            if battery_level - nx.shortest_path_length(self.graph, new_station, destination, weight="battery_reduction") >= 20:
                sub_path = nx.dijkstra_path(self.graph, new_station, destination, weight="distance")
                path.append(sub_path)
                total_distance += nx.path_weight(self.graph, sub_path, "distance")
                total_battery_consumption += nx.path_weight(self.graph, sub_path, "battery_reduction")
                total_charging_time = (total_battery_consumption - battery_level + 20) * 1.25
                spent_time += nx.path_weight(self.graph, sub_path, "distance")
                spent_time += total_charging_time + 80
                return spent_time, total_battery_consumption, total_distance, path, battery_level

            last_source = new_station

    def export_to_qlearning(self, q_values_file="q_values.pkl"):
        q_table = {}

        for u in self.graph.nodes():
            q_table[u] = {}
            for v in self.graph.nodes():
                if u != v:
                    path, distance = self.shortest_path(u, v)
                    q_table[u][v] = -distance  # Initialize Q-value as negative distance for Q-learning

        with open(q_values_file, 'wb') as f:
            pickle.dump(q_table, f)

        print(f"Q-values initialized and exported to {q_values_file}")

    def export_terc_results(self, result_file="terc_results.pkl"):
        terc_results = {
            'charging_stations': self.charging_nodes,
            'graph': self.graph
        }

        with open(result_file, 'wb') as f:
            pickle.dump(terc_results, f)

        print(f"TERC results exported to {result_file}")

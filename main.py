import matplotlib.pyplot as plt
from TQL import QLearning as TQL
from SimpleQLearning import SimpleQLearning
from DQN import DeepQLearning as DQN
from road_network import RandomGraph  # Import road network generator
from TERC2 import TERC2  # Assuming TERC2 is used for initializing Q-values

def plot_metrics(metrics, labels, title, xlabel, ylabel):
    """
    Plot multiple metrics on the same graph.
    
    Parameters:
    - metrics: List of lists containing metric values for each algorithm.
    - labels: List of strings representing the label for each metric.
    - title: String, title of the plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    for metric, label in zip(metrics, labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_evaluate():
    # Step 1: Generate the road network
    print("Generating road network...")
    graph_generator = RandomGraph(num_nodes=100, probability=0.085, seed=42, show_stat=True)
    adjacency_matrix, charging_stations = graph_generator.generate_graph()  # Generate adjacency matrix and charging stations

    # Step 2: Initialize TERC2 for Q-values (if needed)
    print("Initializing TERC2...")
    terc2 = TERC2(adjacency_matrix)
    for station in charging_stations:
        terc2.add_charging_nodes(station)
    
    terc2.export_terc_results("terc_results.pkl")  # Export to Q-learning initialization file
    terc2.export_to_qlearning("q_values.pkl")  # This will create q_values.pkl for Q-learning initialization
    
    # Step 3: Initialize agents with the road network and Q-values
    print("Initializing agents...")
    tql_agent = TQL(adjacency_matrix=adjacency_matrix, num_nodes=100, charging_stations=charging_stations, q_values_file="q_values.pkl")
    simple_q_agent = SimpleQLearning(adjacency_matrix=adjacency_matrix, num_nodes=100, charging_stations=charging_stations)
    dqn_agent = DQN(adjacency_matrix=adjacency_matrix, num_nodes=100, charging_stations=charging_stations)

    # Step 4: Train agents
    print("Training agents...")
    tql_agent.train()
    simple_q_agent.train()
    dqn_agent.train()

    # Step 5: Plot Cumulative Rewards
    print("Plotting cumulative rewards...")
    plot_metrics(
        [tql_agent.epoch_rewards, simple_q_agent.epoch_rewards, dqn_agent.epoch_rewards],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Cumulative Rewards Comparison',
        'Episodes',
        'Cumulative Rewards'
    )

    # Step 6: Plot Travelled Distance
    print("Plotting travelled distances...")
    plot_metrics(
        [tql_agent.epoch_distances, simple_q_agent.epoch_distances, dqn_agent.epoch_distances],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Travelled Distance Comparison',
        'Episodes',
        'Travelled Distance'
    )

    # Step 7: Plot Traveling Time
    print("Plotting traveling times...")
    plot_metrics(
        [tql_agent.epoch_travel_times, simple_q_agent.epoch_travel_times, dqn_agent.epoch_travel_times],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Traveling Time Comparison',
        'Episodes',
        'Traveling Time'
    )

if __name__ == "__main__":
    train_and_evaluate()

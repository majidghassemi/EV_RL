import matplotlib.pyplot as plt
from TQL import QLearning as TQL
from SimpleQLearning import SimpleQLearning
from DQN import DeepQLearning as DQN

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
    # Initialize agents
    tql_agent = TQL()
    simple_q_agent = SimpleQLearning()
    dqn_agent = DQN()

    # Train agents
    tql_agent.train()
    simple_q_agent.train()
    dqn_agent.train()

    # Plot Cumulative Rewards
    plot_metrics(
        [tql_agent.epoch_rewards, simple_q_agent.epoch_rewards, dqn_agent.epoch_rewards],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Cumulative Rewards Comparison',
        'Episodes',
        'Cumulative Rewards'
    )

    # Plot Travelled Distance
    plot_metrics(
        [tql_agent.epoch_distances, simple_q_agent.epoch_distances, dqn_agent.epoch_distances],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Travelled Distance Comparison',
        'Episodes',
        'Travelled Distance'
    )

    # Plot Traveling Time
    plot_metrics(
        [tql_agent.epoch_travel_times, simple_q_agent.epoch_travel_times, dqn_agent.epoch_travel_times],
        ['TQL', 'Simple Q-Learning', 'DQN'],
        'Traveling Time Comparison',
        'Episodes',
        'Traveling Time'
    )

if __name__ == "__main__":
    train_and_evaluate()

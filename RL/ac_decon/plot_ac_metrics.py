import matplotlib.pyplot as plt
import re
import os

def parse_logs(log_file):
    episodes = []
    steps_in_episode = []
    total_rewards = []
    avg_actor_losses = []
    avg_critic_losses = []
    avg_q_values = []

    with open(log_file, 'r') as f:
        for line in f:
            # Match and extract episode information from the log file
            match = re.match(r'^.*Episode (\d+): Steps in Episode: (\d+), Total Reward: ([\d\.-]+), '
                             r'Avg Actor Loss: ([\d\.-]+), Avg Critic Loss: ([\d\.-]+), '
                             r'Avg Q Value: ([\d\.-]+).*$', line)
            if match:
                episodes.append(int(match.group(1)))
                steps_in_episode.append(int(match.group(2)))
                total_rewards.append(float(match.group(3)))
                avg_actor_losses.append(float(match.group(4)))
                avg_critic_losses.append(float(match.group(5)))
                avg_q_values.append(float(match.group(6)))

    return episodes, steps_in_episode, total_rewards, avg_actor_losses, avg_critic_losses, avg_q_values

def plot_metrics(episodes, steps_in_episode, total_rewards, avg_actor_losses, avg_critic_losses, avg_q_values, output_dir):
    plt.figure(figsize=(12, 8))

    # Plot Total Rewards
    plt.subplot(2, 2, 1)
    plt.plot(episodes, total_rewards, label='Total Reward', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.grid(True)

    # Plot Avg Actor Loss
    plt.subplot(2, 2, 2)
    plt.plot(episodes, avg_actor_losses, label='Avg Actor Loss', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Avg Actor Loss')
    plt.title('Average Actor Loss over Episodes')
    plt.grid(True)

    # Plot Avg Critic Loss
    plt.subplot(2, 2, 3)
    plt.plot(episodes, avg_critic_losses, label='Avg Critic Loss', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Avg Critic Loss')
    plt.title('Average Critic Loss over Episodes')
    plt.grid(True)

    # Plot Avg Q Value
    plt.subplot(2, 2, 4)
    plt.plot(episodes, avg_q_values, label='Avg Q Value', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Avg Q Value')
    plt.title('Average Q Value over Episodes')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.show()

if __name__ == "__main__":
    # Directory where the logs are saved
    log_dir = 'C:/Users/aymen/OneDrive/Documents/GitHub/DRL/training_results/logs'
    log_file = os.path.join(log_dir, 'ac_training.log')

    # Output directory for plots
    output_dir = 'C:/Users/aymen/OneDrive/Documents/GitHub/DRL/training_results/plots'

    # Parse the logs to extract metrics
    episodes, steps_in_episode, total_rewards, avg_actor_losses, avg_critic_losses, avg_q_values = parse_logs(log_file)

    # Plot and save the metrics
    plot_metrics(episodes, steps_in_episode, total_rewards, avg_actor_losses, avg_critic_losses, avg_q_values, output_dir)

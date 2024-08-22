import os
import matplotlib.pyplot as plt

# Define the file paths
decon_rewards_path = r'C:\Users\aymen\OneDrive\Documents\GitHub\DRL-1\training_results\plots\ac_decon_reward_data.txt'
vanilla_rewards_path = r'C:\Users\aymen\OneDrive\Documents\GitHub\DRL-1\training_results\plots\ac_vanilla_reward_data.txt'

# Function to load rewards from a file
def load_rewards(file_path):
    with open(file_path, 'r') as file:
        rewards = file.readlines()
    return [float(reward.strip()) for reward in rewards]

# Load the rewards
rewards_decon = load_rewards(decon_rewards_path)
rewards_vanilla = load_rewards(vanilla_rewards_path)

# Plotting the rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards_decon, label='Decon', color='blue')
plt.plot(rewards_vanilla, label='Vanilla', color='orange')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Comparison between Decon and Vanilla')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\aymen\OneDrive\Documents\GitHub\DRL-1\training_results\performance_comparison.png')
plt.show()

import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to parse log file and extract relevant metrics
def parse_log_file(log_file_path):
    data = {
        'epoch': [],
        'nll_tr': [],
        'x_tr_loss': [],
        'a_tr_loss': [],
        'r_tr_loss': []
    }

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'epoch: (\d+),.*?nll_tr: ([\d.]+), x_tr_loss: ([\d.]+), a_tr_loss: ([\d.]+), r_tr_loss: ([\d.]+)', line)
            if match:
                data['epoch'].append(int(match.group(1)))
                data['nll_tr'].append(float(match.group(2)))
                data['x_tr_loss'].append(float(match.group(3)))
                data['a_tr_loss'].append(float(match.group(4)))
                data['r_tr_loss'].append(float(match.group(5)))
    
    return pd.DataFrame(data)

# Function to plot metrics
def plot_metrics(df, save_path):
    epochs = df['epoch'].unique()
    avg_metrics_per_epoch = df.groupby('epoch').mean()

    plt.figure(figsize=(14, 8))

    # Subplot for NLL Training Loss (Epoch Avg)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, avg_metrics_per_epoch['nll_tr'], label='NLL Training Loss (Epoch Avg)', linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Training Loss')
    plt.title('NLL Training Loss Over Epochs\n(MNIST Dataset)')
    plt.legend()

    # Subplot for X Training Loss (Epoch Avg)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, avg_metrics_per_epoch['x_tr_loss'], label='X Training Loss (Epoch Avg)', linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('X Training Loss')
    plt.title('X Training Loss Over Epochs\n(MNIST Dataset)')
    plt.legend()

    # Subplot for A Training Loss (Epoch Avg)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, avg_metrics_per_epoch['a_tr_loss'], label='A Training Loss (Epoch Avg)', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('A Training Loss')
    plt.title('A Training Loss Over Epochs\n(MNIST Dataset)')
    plt.legend()

    # Subplot for R Training Loss (Epoch Avg)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, avg_metrics_per_epoch['r_tr_loss'], label='R Training Loss (Epoch Avg)', linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('R Training Loss')
    plt.title('R Training Loss Over Epochs\n(MNIST Dataset)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Path to the log file
log_file_path = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\training_results\\trainingGaussian.log'

# Parse the log file
df = parse_log_file(log_file_path)

# Define the save path
save_dir = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'mnist_metrics_model_decon_uGaussian.png')

# Plot the metrics
plot_metrics(df, save_path)

print(f'Plots saved to {save_path}')

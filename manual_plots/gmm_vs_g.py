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
        'r_tr_loss': [],
        'elapsed_time': []
    }

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'epoch: (\d+),.*?nll_tr: ([\d.]+), x_tr_loss: ([\d.]+), a_tr_loss: ([\d.]+), r_tr_loss: ([\d.]+), elapsed_time: ([\d.]+)', line)
            if match:
                data['epoch'].append(int(match.group(1)))
                data['nll_tr'].append(float(match.group(2)))
                data['x_tr_loss'].append(float(match.group(3)))
                data['a_tr_loss'].append(float(match.group(4)))
                data['r_tr_loss'].append(float(match.group(5)))
                data['elapsed_time'].append(float(match.group(6)))
    
    return pd.DataFrame(data)

# Function to plot metrics from three dataframes
def plot_comparison(df1, df2, df3, log_names, save_path):
    epochs1 = df1['epoch'].unique()
    avg_metrics_per_epoch1 = df1.groupby('epoch').mean()
    
    epochs2 = df2['epoch'].unique()
    avg_metrics_per_epoch2 = df2.groupby('epoch').mean()

    epochs3 = df3['epoch'].unique()
    avg_metrics_per_epoch3 = df3.groupby('epoch').mean()
    
    plt.figure(figsize=(24, 12))

    # Subplot for NLL Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs1, avg_metrics_per_epoch1['nll_tr'], label=f'NLL Training Loss ({log_names[0]})', linewidth=2, color='blue')
    plt.plot(epochs2, avg_metrics_per_epoch2['nll_tr'], label=f'NLL Training Loss ({log_names[1]})', linewidth=2, color='cyan', linestyle='--')
    plt.plot(epochs3, avg_metrics_per_epoch3['nll_tr'], label=f'NLL Training Loss ({log_names[2]})', linewidth=2, color='navy', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Training Loss')
    plt.title('NLL Training Loss Over Epochs')
    plt.legend()

    # Subplot for X Training Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs1, avg_metrics_per_epoch1['x_tr_loss'], label=f'X Training Loss ({log_names[0]})', linewidth=2, color='green')
    plt.plot(epochs2, avg_metrics_per_epoch2['x_tr_loss'], label=f'X Training Loss ({log_names[1]})', linewidth=2, color='lightgreen', linestyle='--')
    plt.plot(epochs3, avg_metrics_per_epoch3['x_tr_loss'], label=f'X Training Loss ({log_names[2]})', linewidth=2, color='darkgreen', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('X Training Loss')
    plt.title('X Training Loss Over Epochs')
    plt.legend()

    # Subplot for A Training Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs1, avg_metrics_per_epoch1['a_tr_loss'], label=f'A Training Loss ({log_names[0]})', linewidth=2, color='red')
    plt.plot(epochs2, avg_metrics_per_epoch2['a_tr_loss'], label=f'A Training Loss ({log_names[1]})', linewidth=2, color='salmon', linestyle='--')
    plt.plot(epochs3, avg_metrics_per_epoch3['a_tr_loss'], label=f'A Training Loss ({log_names[2]})', linewidth=2, color='darkred', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('A Training Loss')
    plt.title('A Training Loss Over Epochs')
    plt.legend()

    # Subplot for R Training Loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs1, avg_metrics_per_epoch1['r_tr_loss'], label=f'R Training Loss ({log_names[0]})', linewidth=2, color='purple')
    plt.plot(epochs2, avg_metrics_per_epoch2['r_tr_loss'], label=f'R Training Loss ({log_names[1]})', linewidth=2, color='violet', linestyle='--')
    plt.plot(epochs3, avg_metrics_per_epoch3['r_tr_loss'], label=f'R Training Loss ({log_names[2]})', linewidth=2, color='indigo', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('R Training Loss')
    plt.title('R Training Loss Over Epochs')
    plt.legend()

    # Subplot for Elapsed Time
    plt.subplot(2, 3, 5)
    plt.plot(epochs1, avg_metrics_per_epoch1['elapsed_time'], label=f'Elapsed Time ({log_names[0]})', linewidth=2, color='orange')
    plt.plot(epochs2, avg_metrics_per_epoch2['elapsed_time'], label=f'Elapsed Time ({log_names[1]})', linewidth=2, color='gold', linestyle='--')
    plt.plot(epochs3, avg_metrics_per_epoch3['elapsed_time'], label=f'Elapsed Time ({log_names[2]})', linewidth=2, color='darkorange', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Elapsed Time')
    plt.title('Elapsed Time Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Paths to the log files
log_file_path1 = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\training_results\\mnist.log'
log_file_path2 = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\training_results\\trainingGaussian.log'
log_file_path3 = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\training_results\\traininguBernoulli.log'

# Parse the log files
df1 = parse_log_file(log_file_path1)
df2 = parse_log_file(log_file_path2)
df3 = parse_log_file(log_file_path3)

# Define the save path
save_dir = 'C:\\Users\\aymen\\OneDrive\\Documents\\GitHub\\DRL-1\\plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'mnist_comparison_upriors.png')

# Plot the comparison metrics
plot_comparison(df1, df2, df3, ['uGMM', 'uGaussian', 'uBernoulli'], save_path)

print(f'Comparison plots saved to {save_path}')

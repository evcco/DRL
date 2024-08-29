#########################
## Author: Chaochao Lu ##
#########################
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model_decon import Model_Decon
from decon_ac_train import AC_Decon
from data_handler import DataHandler
import configs

def compare_ac_models(sess, opts, data, model_class, ac_class):
    # Initialize old and new models
    old_opts = opts.copy()
    new_opts = opts.copy()

    # Assuming you have saved checkpoints for the old model
    old_opts['model_checkpoint'] = './path_to_old_model_checkpoint'
    new_opts['model_checkpoint'] = './path_to_new_model_checkpoint'

    print('Initializing old model ...')
    old_model = model_class(sess, old_opts)
    old_ac = ac_class(sess, old_opts, old_model)

    print('Initializing new model ...')
    new_model = model_class(sess, new_opts)
    new_ac = ac_class(sess, new_opts, new_model)

    # Placeholder for metrics
    old_rewards, new_rewards = [], []
    old_losses, new_losses = [], []

    # Number of evaluation runs
    eval_runs = 100

    for i in range(eval_runs):
        # Simulate one episode
        old_reward, old_loss = old_ac.evaluate(data)
        new_reward, new_loss = new_ac.evaluate(data)

        # Collect metrics
        old_rewards.append(old_reward)
        new_rewards.append(new_reward)
        old_losses.append(old_loss)
        new_losses.append(new_loss)

    # Compute average metrics
    avg_old_reward = np.mean(old_rewards)
    avg_new_reward = np.mean(new_rewards)
    avg_old_loss = np.mean(old_losses)
    avg_new_loss = np.mean(new_losses)

    print(f'Old Model: Avg Reward: {avg_old_reward}, Avg Loss: {avg_old_loss}')
    print(f'New Model: Avg Reward: {avg_new_reward}, Avg Loss: {avg_new_loss}')

    # Plot rewards
    plt.figure()
    plt.plot(range(eval_runs), old_rewards, label='Old Model Rewards')
    plt.plot(range(eval_runs), new_rewards, label='New Model Rewards')
    plt.legend()
    plt.title('Comparison of Rewards')
    plt.show()

    # Plot losses
    plt.figure()
    plt.plot(range(eval_runs), old_losses, label='Old Model Losses')
    plt.plot(range(eval_runs), new_losses, label='New Model Losses')
    plt.legend()
    plt.title('Comparison of Losses')
    plt.show()

def main():
    opts = configs.model_config

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    print('Processing data ...')
    data = DataHandler(opts)

    print('Comparing old and new AC_Decon models ...')
    compare_ac_models(sess, opts, data, Model_Decon, AC_Decon)

if __name__ == "__main__":
    main()

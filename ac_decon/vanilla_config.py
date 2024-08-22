#########################
## Author: Chaochao Lu ##
#########################
from collections import OrderedDict
import tensorflow as tf

########################################################################################################################
########################################## Vanilla AC Model Configuration ##############################################
########################################################################################################################

vanilla_config = OrderedDict()

########################################## Data and Model Path Configuration ###########################################

vanilla_config['work_dir'] = './training_results'
vanilla_config['data_dir'] = './dataset'
vanilla_config['training_data'] = './mnist_training_data.npz'
vanilla_config['validation_data'] = './mnist_validation_data.npz'
vanilla_config['testing_data'] = './mnist_testing_data.npz'
vanilla_config['model_checkpoint'] = './training_results/model_checkpoints/model_vanilla'
vanilla_config['policy_checkpoint'] = './training_results/policy_checkpoints/policy_vanilla'

########################################################################################################################

vanilla_config['dataset'] = 'mnist'

########################################################################################################################

vanilla_config['seed'] = 123
vanilla_config['lr'] = 0.0001

vanilla_config['is_conv'] = True
vanilla_config['gated'] = False

vanilla_config['is_restored'] = False
vanilla_config['model_checkpoint'] = None
vanilla_config['epoch_start'] = 0
vanilla_config['counter_start'] = 0

vanilla_config['init_std'] = 0.0099999
vanilla_config['init_bias'] = 0.0
vanilla_config['filter_size'] = 5

vanilla_config['a_range'] = 2

vanilla_config['z_dim'] = 50
vanilla_config['x_dim'] = 784  # 28 x 28
vanilla_config['a_dim'] = 1
vanilla_config['u_dim'] = 1  # Not used in Vanilla AC, but kept for consistency
vanilla_config['a_latent_dim'] = 100
vanilla_config['r_dim'] = 1
vanilla_config['r_latent_dim'] = 100
vanilla_config['mask_dim'] = 1
vanilla_config['lstm_dim'] = 100
vanilla_config['mnist_dim'] = 28
vanilla_config['mnist_channel'] = 1

vanilla_config['batch_size'] = 128
vanilla_config['nsteps'] = 5
vanilla_config['sample_num'] = 5
vanilla_config['epoch_num'] = 400

vanilla_config['save_every_epoch'] = 10
vanilla_config['plot_every'] = 500
vanilla_config['inference_model_type'] = 'LR'
vanilla_config['lstm_dropout_prob'] = 0.
vanilla_config['recons_cost'] = 'l2sq'
vanilla_config['anneal'] = 1

vanilla_config['work_dir'] = './training_results'
vanilla_config['data_dir'] = './dataset'

vanilla_config['model_bn_is_training'] = True

########################################################################################################################
########################################## AC Configuration ############################################################
########################################################################################################################

vanilla_config['replay_memory_capacity'] = int(1e5)
vanilla_config['tau'] = 1e-2
vanilla_config['gamma'] = 0.99
vanilla_config['l2_reg_critic'] = 1e-6
vanilla_config['lr_critic'] = 1e-3
vanilla_config['lr_decay'] = 1
vanilla_config['l2_reg_actor'] = 1e-6
vanilla_config['lr_actor'] = 1e-3
vanilla_config['dropout_rate'] = 0

vanilla_config['policy_net_layers'] = [300, 300]
vanilla_config['policy_net_outlayers'] = [[1, tf.nn.tanh],
                                          [1, tf.nn.softplus]]
vanilla_config['value_net_layers'] = [300, 300]
vanilla_config['value_net_outlayers'] = [[1, None],
                                         [1, tf.nn.softplus]]

vanilla_config['episode_num'] = 2000
vanilla_config['episode_start'] = 0
vanilla_config['save_every_episode'] = 100
vanilla_config['max_steps_in_episode'] = 200
vanilla_config['train_every'] = 1
vanilla_config['mini_batch_size'] = 128
vanilla_config['u_sample_size'] = 200  # Not used in Vanilla AC, but kept for consistency

vanilla_config['final_reward'] = 0

vanilla_config['policy_test_episode_num'] = 100

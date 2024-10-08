#########################
## Author: Chaochao Lu ##
#########################
from collections import OrderedDict
import tensorflow as tf

########################################################################################################################
########################################## Full Model Configuration ####################################################
########################################################################################################################

model_config = OrderedDict()

########################################## Data and Model Path Configuration ###########################################

model_config['work_dir'] = './training_results'
model_config['data_dir'] = './dataset'
model_config['training_data'] = './mnist_training_data.npz'
model_config['validation_data'] = './mnist_validation_data.npz'
model_config['testing_data'] = './mnist_testing_data.npz'
model_config['model_checkpoint'] = './training_results/model_checkpoints/model_con'
model_config['policy_checkpoint'] = './training_results/policy_checkpoints/policy_con'

########################################################################################################################

model_config['dataset'] = 'mnist'

########################################################################################################################

# General configurations
model_config['seed'] = 123
model_config['lr'] = 0.0001

model_config['is_conv'] = True
model_config['gated'] = True

model_config['is_restored'] = False
model_config['model_checkpoint'] = None
model_config['epoch_start'] = 0
model_config['counter_start'] = 0

model_config['init_std'] = 0.0099999
model_config['init_bias'] = 0.0
model_config['filter_size'] = 5

model_config['a_range'] = 2

model_config['z_dim'] = 50
model_config['x_dim'] = 784  # 28 x 28
model_config['a_dim'] = 1
model_config['u_dim'] = 1
model_config['a_latent_dim'] = 100
model_config['r_dim'] = 1
model_config['r_latent_dim'] = 100
model_config['mask_dim'] = 1
model_config['lstm_dim'] = 100
model_config['mnist_dim'] = 28
model_config['mnist_channel'] = 1

model_config['batch_size'] = 128
model_config['nsteps'] = 5
model_config['sample_num'] = 5
model_config['epoch_num'] = 400

model_config['save_every_epoch'] = 10
model_config['plot_every'] = 500
model_config['inference_model_type'] = 'LR'
model_config['lstm_dropout_prob'] = 0.
model_config['recons_cost'] = 'l2sq'
model_config['anneal'] = 1

model_config['work_dir'] = './training_results'
model_config['data_dir'] = '/scratch/cl641/CausalRL/bcdr_beta/data_prep'

########################################################################################################################
########################################## Network Layer Configurations ################################################
########################################################################################################################

# p(x|z,u) network configuration
model_config['pxgz_net_layers'] = [100, 100]
model_config['pxgz_net_outlayers'] = []
model_config['pxgu_net_layers'] = [100, 100]
model_config['pxgu_net_outlayers'] = []
model_config['pxgzu_net_layers'] = [300, 300]
model_config['pxgzu_net_outlayers'] = [[model_config['x_dim'], tf.nn.sigmoid]]

# p(a|z,u) network configuration
model_config['pagz_net_layers'] = [100, 100]
model_config['pagz_net_outlayers'] = []
model_config['pagu_net_layers'] = [100, 100]
model_config['pagu_net_outlayers'] = []
model_config['pagzu_net_layers'] = [300, 300]
model_config['pagzu_net_outlayers'] = [[model_config['a_dim'], tf.nn.tanh]]

# p(r|z,a,u) network configuration
model_config['prgz_net_layers'] = [100, 100]
model_config['prgz_net_outlayers'] = []
model_config['prga_net_layers'] = [100, 100]
model_config['prga_net_outlayers'] = []
model_config['prgu_net_layers'] = [100, 100]
model_config['prgu_net_outlayers'] = []
model_config['prgzau_net_layers'] = [300, 300]
model_config['prgzau_net_outlayers'] = [[model_config['r_dim'], tf.nn.sigmoid]]

# p(z|z,a) network configuration
model_config['pzgz_net_layers'] = [100]
model_config['pzgz_net_outlayers'] = []
model_config['pzga_net_layers'] = [model_config['a_latent_dim']]
model_config['pzga_net_outlayers'] = []
model_config['pzgza_net_layers'] = [100]
model_config['pzgza_net_outlayers'] = []
model_config['pzgza_mu_net_layers'] = []
model_config['pzgza_mu_net_outlayers'] = [[model_config['z_dim'], None]]
model_config['pzgza_sigma_net_layers'] = []
model_config['pzgza_sigma_net_outlayers'] = [[model_config['z_dim'], tf.nn.softplus]]

# q(z|x,a,r) network configuration
model_config['qzgx_net_layers'] = [300, 300, 100]
model_config['qzgx_net_outlayers'] = []
model_config['qzga_net_layers'] = [100, model_config['a_latent_dim']]
model_config['qzga_net_outlayers'] = []
model_config['qzgr_net_layers'] = [100, model_config['r_latent_dim']]
model_config['qzgr_net_outlayers'] = []
model_config['qzgxar_net_layers'] = [100]
model_config['qzgxar_net_outlayers'] = [[100, None]]

# q(u|x,a,r) network configuration
model_config['qugx_net_layers'] = [300, 300, 100]
model_config['qugx_net_outlayers'] = []
model_config['quga_net_layers'] = [100, model_config['a_latent_dim']]
model_config['quga_net_outlayers'] = []
model_config['qugr_net_layers'] = [100, model_config['r_latent_dim']]
model_config['qugr_net_outlayers'] = []
model_config['qugxar_net_layers'] = [100]
model_config['qugxar_net_outlayers'] = [[100, None]]
model_config['qugh_net_layers'] = [100]
model_config['qugh_net_outlayers'] = [[model_config['u_dim'], None]]

########################################################################################################################
########################################## Actor-Critic Configuration ##################################################
########################################################################################################################

model_config['replay_memory_capacity'] = int(1e5)
model_config['tau'] = 1e-2
model_config['gamma'] = 0.99
model_config['l2_reg_critic'] = 1e-6
model_config['lr_critic'] = 1e-3
model_config['lr_decay'] = 1
model_config['l2_reg_actor'] = 1e-6
model_config['lr_actor'] = 1e-3
model_config['dropout_rate'] = 0

model_config['policy_net_layers'] = [300, 300]
model_config['policy_net_outlayers'] = [[1, tf.nn.tanh]]
model_config['value_net_layers'] = [300, 300]
model_config['value_net_outlayers'] = [[1, None]]

model_config['episode_num'] = 2000
model_config['episode_start'] = 0
model_config['save_every_episode'] = 100
model_config['max_steps_in_episode'] = 200
model_config['train_every'] = 1
model_config['mini_batch_size'] = 128
model_config['u_sample_size'] = 200

model_config['final_reward'] = 0
model_config['policy_test_episode_num'] = 100

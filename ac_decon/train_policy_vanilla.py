import numpy as np
import tensorflow as tf
import os
import random
from collections import deque, OrderedDict
from data_handler import DataHandler

class VanillaAC(object):
    def __init__(self, sess, opts):
        self.sess = sess
        self.opts = opts
        self.saver = None

        np.random.seed(self.opts['seed'])

        # Placeholders
        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.opts['x_dim']])
        self.action_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']])
        self.reward_ph = tf.placeholder(tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(tf.float32, shape=[None, self.opts['x_dim']])
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None])

        self.replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])

        # Build networks and training operations
        self._build_networks()
        self._build_training_ops()

        # TensorFlow saver for saving the model
        self.saver = tf.train.Saver(max_to_keep=50)

    def _build_networks(self):
        # Actor network
        with tf.variable_scope('actor'):
            self.action_mu, self.action_sigma = self._actor_net(self.state_ph)
        # Target Actor network
        with tf.variable_scope('target_actor'):
            self.target_action_mu, self.target_action_sigma = self._actor_net(self.next_state_ph)

        # Critic network
        with tf.variable_scope('critic'):
            self.q_value = self._critic_net(self.state_ph, self.action_ph)
        # Target Critic network
        with tf.variable_scope('target_critic'):
            self.target_q_value = self._critic_net(self.next_state_ph, self.target_action_mu)

        # Network variables
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Update target networks
        self.update_targets_op = []
        self.update_targets_op += self._update_target_network('target_actor', 'actor')
        self.update_targets_op += self._update_target_network('target_critic', 'critic')
        self.update_targets_op = tf.group(*self.update_targets_op)

    def _update_target_network(self, target_scope, main_scope):
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=main_scope)
        update_ops = []
        for t_var, m_var in zip(target_vars, main_vars):
            update_op = t_var.assign(self.opts['tau'] * m_var + (1 - self.opts['tau']) * t_var)
            update_ops.append(update_op)
        return update_ops

    def _build_training_ops(self):
        # Critic loss and optimization
        self.targets = self.reward_ph + self.opts['gamma'] * self.is_not_terminal_ph * self.target_q_value
        self.td_errors = self.targets - self.q_value
        self.critic_loss = tf.reduce_mean(tf.square(self.td_errors))
        self.critic_train_op = tf.train.AdamOptimizer(self.opts['lr_critic']).minimize(self.critic_loss, var_list=self.critic_vars)

        # Actor loss and optimization
        with tf.variable_scope('critic', reuse=True):
            q_value_for_actor = self._critic_net(self.state_ph, self.action_mu)
        self.actor_loss = -tf.reduce_mean(q_value_for_actor)
        self.actor_train_op = tf.train.AdamOptimizer(self.opts['lr_actor']).minimize(self.actor_loss, var_list=self.actor_vars)

    def _actor_net(self, state):
        hidden = tf.layers.dense(state, 300, activation=tf.nn.relu)
        hidden = tf.layers.dense(hidden, 300, activation=tf.nn.relu)
        action_mu = tf.layers.dense(hidden, self.opts['a_dim'], activation=tf.nn.tanh)
        action_sigma = tf.layers.dense(hidden, self.opts['a_dim'], activation=tf.nn.softplus)
        return action_mu, action_sigma

    def _critic_net(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        hidden = tf.layers.dense(state_action, 300, activation=tf.nn.relu)
        hidden = tf.layers.dense(hidden, 300, activation=tf.nn.relu)
        q_value = tf.layers.dense(hidden, 1, activation=None)
        return q_value

    def create_env(self):
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        action_mu, action_sigma = self.sess.run([self.action_mu, self.action_sigma], feed_dict={self.state_ph: state})
        action = np.random.normal(action_mu, action_sigma)
        return np.clip(action, -self.opts['a_range'], self.opts['a_range'])

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    def train(self, data):
        total_steps = 0
        total_episode = 0

        reward_que = deque(maxlen=100)
        reward_list = []

        self.create_env()

        work_dir = self.opts.get('work_dir', './training_results')
        plots_dir = os.path.join(work_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        checkpoints_dir = os.path.join(work_dir, 'policy_checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        for episode in range(self.opts['episode_num']):
            reward_data_path = os.path.join(plots_dir, 'vanilla_ac_reward_data.txt')
            with open(reward_data_path, 'a+') as f:

                if episode > self.opts['episode_start'] and episode % self.opts['save_every_episode'] == 0:
                    reward_filename = 'vanilla_ac_reward_plot_epoch_{:d}.png'.format(episode)
                    # Save reward plots (implement save_reward_plots as needed)
                    # save_reward_plots(self.opts, reward_list, reward_filename)
                    self.saver.save(self.sess, os.path.join(checkpoints_dir, 'policy_vanilla_ac'), global_step=total_episode)

                total_reward = 0
                steps_in_episode = 0

                # Randomly select a batch of initial states from the training data
                tr_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
                tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
                state = np.reshape(data.x_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['x_dim']])

                for step in range(self.opts['max_steps_in_episode']):
                    action = self.choose_action(state)
                    next_state = np.reshape(data.x_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['x_dim']])  # Randomly pick next state
                    reward = np.random.rand()  # Dummy reward
                    done = step == self.opts['max_steps_in_episode'] - 1

                    # Add to memory
                    self.add_to_memory((state, action, reward, next_state, 0.0 if done else 1.0))

                    # Update parameters using mini-batch of experience
                    if total_steps % self.opts['train_every'] == 0 and len(self.replay_memory) >= self.opts['mini_batch_size']:
                        mini_batch = self.sample_from_memory(self.opts['mini_batch_size'])
                        states, actions, rewards, next_states, is_not_terminals = zip(*mini_batch)

                        # Print shapes before reshaping
                        print("Before Reshape - States shape:", np.array(states).shape)

                        # Check the shape before reshaping
                        print("Raw states shape:", np.array(states).shape)

                        # Ensure states and next_states are flattened and reshaped correctly
                        states = np.array(states).reshape([self.opts['mini_batch_size'], self.opts['x_dim']])
                        next_states = np.array(next_states).reshape([self.opts['mini_batch_size'], self.opts['x_dim']])

                        print("Reshaped states shape:", states.shape)
                        print("Reshaped next_states shape:", next_states.shape)

                        self.sess.run(self.critic_train_op, feed_dict={
                            self.state_ph: states,
                            self.action_ph: actions,
                            self.reward_ph: rewards,
                            self.next_state_ph: next_states,
                            self.is_not_terminal_ph: is_not_terminals
                        })

                        self.sess.run(self.actor_train_op, feed_dict={
                            self.state_ph: states
                        })

                        self.sess.run(self.update_targets_op)

                    state = next_state
                    total_reward += reward
                    total_steps += 1
                    steps_in_episode += 1

                    if done:
                        break

                total_episode += 1

                print('Episode: {:d}, Steps in Episode: {:d}, Total Reward: {:f}'.format(episode, steps_in_episode, total_reward))

                reward_que.append(total_reward)
                reward_list.append(np.mean(reward_que))

                f.write('{:f}\n'.format(np.mean(reward_que)))

        self.saver.save(self.sess, os.path.join(checkpoints_dir, 'policy_vanilla_ac'), global_step=total_episode)

## no need to call the config file, just define the dictionary here
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
vanilla_config['a_latent_dim'] = 100
vanilla_config['r_dim'] = 1
vanilla_config['r_latent_dim'] = 100
vanilla_config['mask_dim'] = 1
vanilla_config['lstm_dim'] = 100
vanilla_config['mnist_dim'] = 28 ## update when using MNIST FROM TF
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

vanilla_config['final_reward'] = 0

vanilla_config['policy_test_episode_num'] = 100

def main():
    opts = vanilla_config
    tf.reset_default_graph()
    sess = tf.Session()
    data = DataHandler(opts)
    ac = VanillaAC(sess, vanilla_config)
    ac.train(data)

if __name__ == "__main__":
    main()

from collections import deque
import numpy as np
import tensorflow as tf
import os
import random
import time
from utils import *

class SAC_Decon(object):

    def __init__(self, sess, opts, model):
        self.sess = sess
        self.opts = opts
        self.model = model
        self.saver = None

        np.random.seed(0)

        # Placeholders for inputs
        self.x = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['x_dim']])
        self.z = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['z_dim']])
        self.a = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['a_dim']])
        self.r = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['r_dim']])
        self.u = tf.placeholder(tf.float32, shape=[self.opts['u_sample_size'], self.opts['batch_size'], self.opts['u_dim']])

        # Initial hidden state z_0
        _, z_mu_init, _ = self.model.q_z_g_z_x_a_r(self.x, self.a, self.r)
        self.z_init = z_mu_init

        # Initialization of u
        self.u_init = tf.random_normal((self.opts['u_sample_size'], self.opts['batch_size'], self.opts['u_dim']),
                                       0., 1., dtype=tf.float32)

        # Compute z_next given z_current and a_current
        self.z_mu_next, _ = self.model.p_z_g_z_a(self.z, self.a)
        self.z_next = self.z_mu_next

        self.reward = None
        self.r_next = self.compute_r_g_cu()

        # Define placeholders for SAC
        self.z_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.a_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']])
        self.r_ph = tf.placeholder(tf.float32, shape=[None])
        self.z_next_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None])
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

        # Replay buffer
        self.replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])

        self.episodes = tf.Variable(0.0, trainable=False)
        self.episode_increase_op = self.episodes.assign_add(1)

        # Soft Actor-Critic architecture
        self.build_sac_network()
        self.build_sac_loss()
        self.update_target_value_network()

    def build_sac_network(self):
        # Define the policy network
        with tf.variable_scope('policy_net'):
            self.a_mu, self.a_sigma = self.actor_net(self.z_ph, reuse=False, is_training=self.is_training_ph)
            self.a_samples = self.sample_action(self.a_mu, self.a_sigma)

        # Define the twin Q-networks
        with tf.variable_scope('q_net_1'):
            self.q1 = self.critic_net(self.z_ph, self.a_ph, reuse=False, is_training=self.is_training_ph)
        with tf.variable_scope('q_net_2'):
            self.q2 = self.critic_net(self.z_ph, self.a_ph, reuse=False, is_training=self.is_training_ph)

        # Define the value network
        with tf.variable_scope('value_net'):
            self.v = self.value_net(self.z_ph, reuse=False, is_training=self.is_training_ph)
        
        # Define the target value network
        with tf.variable_scope('target_value_net'):
            self.v_target = self.value_net(self.z_next_ph, reuse=False, is_training=self.is_training_ph)
            self.v_target = tf.stop_gradient(self.v_target)

        # Collect variables
        self.policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_net')
        self.q1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_net_1')
        self.q2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_net_2')
        self.value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value_net')
        self.target_value_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_value_net')
        
        # Entropy term
        self.alpha = tf.Variable(0.2, trainable=True, name="entropy_coefficient")

    def build_sac_loss(self):
        # Critic losses
        q_target = tf.stop_gradient(self.r_ph + self.opts['gamma'] * self.is_not_terminal_ph * self.v_target)
        q1_loss = tf.losses.mean_squared_error(labels=q_target, predictions=self.q1)
        q2_loss = tf.losses.mean_squared_error(labels=q_target, predictions=self.q2)

        # Actor loss: maximize Q with entropy regularization
        log_pi = self.compute_log_pi(self.a_mu, self.a_sigma, self.a_samples)
        min_q = tf.minimum(self.q1, self.q2)
        actor_loss = tf.reduce_mean(self.alpha * log_pi - min_q)

        # Value loss: minimize the difference between value network and expected Q
        v_target = tf.stop_gradient(min_q - self.alpha * log_pi)
        value_loss = tf.losses.mean_squared_error(labels=v_target, predictions=self.v)

        # Entropy coefficient loss
        alpha_loss = -tf.reduce_mean(self.alpha * (log_pi + self.opts['target_entropy']))

        # Optimizers
        self.actor_optimizer = tf.train.AdamOptimizer(self.opts['lr_actor']).minimize(actor_loss, var_list=self.policy_vars)
        self.q1_optimizer = tf.train.AdamOptimizer(self.opts['lr_critic']).minimize(q1_loss, var_list=self.q1_vars)
        self.q2_optimizer = tf.train.AdamOptimizer(self.opts['lr_critic']).minimize(q2_loss, var_list=self.q2_vars)
        self.value_optimizer = tf.train.AdamOptimizer(self.opts['lr_value']).minimize(value_loss, var_list=self.value_vars)
        self.alpha_optimizer = tf.train.AdamOptimizer(self.opts['lr_alpha']).minimize(alpha_loss, var_list=[self.alpha])

    def update_target_value_network(self):
        update_ops = []
        for target_var, var in zip(self.target_value_vars, self.value_vars):
            update_ops.append(target_var.assign(self.opts['tau'] * var + (1 - self.opts['tau']) * target_var))
        self.update_targets_op = tf.group(*update_ops)

    def sample_action(self, mu, sigma):
        eps = tf.random_normal(tf.shape(mu))
        action = mu + eps * tf.exp(sigma)
        return tf.tanh(action) * self.opts['a_range']

    def compute_log_pi(self, mu, sigma, action):
        dist = tf.distributions.Normal(loc=mu, scale=sigma)
        log_pi = dist.log_prob(action) - tf.reduce_sum(tf.log(1 - action ** 2 + 1e-6), axis=1)
        return log_pi

    def actor_net(self, z, reuse, is_training):
        mu, sigma = ac_fc_net(self.opts, z, self.opts['policy_net_layers'],
                              self.opts['policy_net_outlayers'], 'policy_net',
                              reuse=reuse, is_training=is_training, trainable=True)
        mu = mu * 2
        return mu, sigma

    def critic_net(self, z, a, reuse, is_training):
        z_a = tf.concat([z, a], axis=1)
        mu, sigma = ac_fc_net(self.opts, z_a, self.opts['value_net_layers'],
                              self.opts['value_net_outlayers'], 'value_net',
                              reuse=reuse, is_training=is_training, trainable=True)
        return mu, sigma

    def value_net(self, z, reuse, is_training):
        mu, sigma = ac_fc_net(self.opts, z, self.opts['value_net_layers'],
                              self.opts['value_net_outlayers'], 'value_net',
                              reuse=reuse, is_training=is_training, trainable=True)
        return mu

    def compute_z_init(self, x, a, r):
        z_init = self.sess.run(self.z_init, feed_dict={self.x: x, self.a: a, self.r: r})
        return z_init

    def compute_u_init(self, x, a, r):
        u_init = self.sess.run(self.u_init, feed_dict={self.x: x, self.a: a, self.r: r})
        return u_init

    def compute_r_g_cu(self):
        self.reward = []
        for i in range(self.opts['u_sample_size']):
            reward_mu, _ = self.model.p_r_g_z_a_u(self.z, self.a,
                                                  tf.reshape(self.u[i], [self.opts['batch_size'],
                                                                         self.opts['u_dim']]))
            self.reward.append(reward_mu)
        return tf.reduce_mean(self.reward)

    def step(self, z, a, x, u):
        z_next_value, r_next_value, reward_samples = \
            self.sess.run([self.z_next, self.r_next, self.reward],
                          feed_dict={self.z: z, self.a: a, self.x: x, self.u: u})

        done = np.abs(r_next_value - self.opts['final_reward']) == 0

        return z_next_value, np.reshape(r_next_value, self.opts['r_dim'])[0], np.reshape(done, 1)[0], reward_samples

    def choose_action(self, z, is_training):
        action = self.sess.run(self.a_samples, feed_dict={self.z_ph: z, self.is_training_ph: is_training})
        return action

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

        for episode in range(self.opts['episode_num']):
            total_reward = 0
            steps_in_episode = 0

            # Sample initial state
            tr_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
            tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
            tr_x_init = np.reshape(data.x_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['x_dim']])
            tr_a_init = np.reshape(data.a_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['a_dim']])
            tr_r_init = np.reshape(data.r_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['r_dim']])

            z = self.compute_z_init(tr_x_init, tr_a_init, tr_r_init)
            u_est = self.compute_u_init(tr_x_init, tr_a_init, tr_r_init)

            for step in range(self.opts['max_steps_in_episode']):
                action = self.choose_action(z, is_training=True)

                z_next, reward, done, reward_samples = self.step(z, action, tr_x_init, u_est)

                total_reward += reward

                self.add_to_memory((np.reshape(z, self.opts['z_dim']), np.reshape(action, self.opts['a_dim']), reward,
                                    np.reshape(z_next, self.opts['z_dim']), 0.0 if done else 1.0))

                # Update parameters using mini-batch of experience
                if total_steps % self.opts['train_every'] == 0 and len(self.replay_memory) >= self.opts['mini_batch_size']:
                    mini_batch = self.sample_from_memory(self.opts['mini_batch_size'])

                    _, _, _, _, _ = self.sess.run(
                        [self.actor_optimizer, self.q1_optimizer, self.q2_optimizer, self.value_optimizer, self.alpha_optimizer],
                        feed_dict={
                            self.z_ph: np.asarray([elem[0] for elem in mini_batch]),
                            self.a_ph: np.asarray([elem[1] for elem in mini_batch]),
                            self.r_ph: np.asarray([elem[2] for elem in mini_batch]),
                            self.z_next_ph: np.asarray([elem[3] for elem in mini_batch]),
                            self.is_not_terminal_ph: np.asarray([elem[4] for elem in mini_batch]),
                            self.is_training_ph: True
                        }
                    )

                    _ = self.sess.run(self.update_targets_op)

                z = z_next
                total_steps += 1
                steps_in_episode += 1

                if done:
                    _ = self.sess.run(self.episode_increase_op)
                    break

            total_episode += 1

            print('Episode: {:d}, Steps in Episode: {:d}, Total Reward: {:f}'.format(episode, steps_in_episode, total_reward))

            reward_que.append(total_reward)
            reward_list.append(np.mean(reward_que))

            reward_data_path = os.path.join(self.opts['work_dir'], 'plots', 'sac_decon_reward_data.txt')
            with open(reward_data_path, 'a+') as f:
                f.write('{:f}\n'.format(np.mean(reward_que)))

            if episode > self.opts['episode_start'] and episode % self.opts['save_every_episode'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'policy_checkpoints', 'policy_decon'), global_step=total_episode)

        self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'policy_checkpoints', 'policy_decon'), global_step=total_episode)

    def create_env(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(self.model_all_vars)

        default_checkpoint_path = "./model_decon_uGaussian"
        checkpoint_path = self.opts.get('model_checkpoint', default_checkpoint_path)
        
        if checkpoint_path and os.path.exists(checkpoint_path + ".index"):
            self.saver.restore(self.sess, checkpoint_path)
            print(f"Model restored from checkpoint: {checkpoint_path}")
        else:
            print(f"No valid checkpoint found at {checkpoint_path}. Initializing variables.")

        self.saver = tf.train.Saver(max_to_keep=50)

from collections import deque
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.mixture import BayesianGaussianMixture
from utils import *


class AC_Decon(object):

    def __init__(self, sess, opts, model):
        self.sess = sess
        self.opts = opts
        self.model = model
        self.saver = None

        np.random.seed(0)

        self.x = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['x_dim']])
        self.z = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['z_dim']])
        self.a = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['a_dim']])
        self.r = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['r_dim']])
        self.u = tf.placeholder(tf.float32, shape=[self.opts['u_sample_size'],
                                                   self.opts['batch_size'], self.opts['u_dim']])

        _, z_mu_init, z_cov_init = self.model.q_z_g_z_x_a_r(self.x, self.a, self.r)
        self.z_init = z_mu_init

        # Using Bayesian Gaussian Mixture Model for `u` initialization
        self.u_init = self.initialize_u_with_gmm()

        self.z_mu_next, z_cov_next = self.model.p_z_g_z_a(self.z, self.a)
        self.z_next = self.z_mu_next

        self.reward = None
        self.r_next = self.compute_r_g_cu()

        self.model_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        self.z_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.a_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']])
        self.r_ph = tf.placeholder(tf.float32, shape=[None])
        self.z_next_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None])

        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

        self.episodes = tf.Variable(0.0, trainable=False)
        self.episode_increase_op = self.episodes.assign_add(1)

        self.replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])

        with tf.variable_scope('actor_net'):
            self.a_mu, self.a_sigma = self.actor_net(self.z_ph, False, self.is_training_ph, True)
        eps = tf.random_normal((1, self.opts['a_dim']), 0., 1., dtype=tf.float32)
        self.a_samples = tf.clip_by_value(self.a_mu + tf.multiply(eps, tf.sqrt(1e-8 + self.a_sigma)),
                                          -self.opts['a_range'], self.opts['a_range'])

        with tf.variable_scope('target_actor_net', reuse=False):
            self.target_a_mu, self.target_a_sigma = self.actor_net(self.z_next_ph, False, self.is_training_ph, False)
            self.target_a_mu = tf.stop_gradient(self.target_a_mu)
            self.target_a_sigma = tf.stop_gradient(self.target_a_sigma)

        with tf.variable_scope('critic_net'):
            self.q_a_mu, self.q_a_sigma = self.critic_net(self.z_ph, self.a_ph, False, self.is_training_ph, True)
            self.q_sa_mu, self.q_sa_sigma = self.critic_net(self.z_ph, self.a_mu, True, self.is_training_ph, True)

        with tf.variable_scope('target_critic_net', reuse=False):
            self.q_next_mu, self.q_next_sigma = \
                self.critic_net(self.z_next_ph, self.target_a_mu, False, self.is_training_ph, False)
            self.q_next_mu = tf.stop_gradient(self.q_next_mu)
            self.q_next_sigma = tf.stop_gradient(self.q_next_sigma)

        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_net')

        update_target_ops = []
        for i, target_actor_var in enumerate(target_actor_vars):
            update_target_actor_op = target_actor_var.assign(self.opts['tau'] * actor_vars[i] +
                                                             (1 - self.opts['tau']) * target_actor_var)
            update_target_ops.append(update_target_actor_op)

        for i, target_critic_var in enumerate(target_critic_vars):
            update_target_critic_op = target_critic_var.assign(self.opts['tau'] * critic_vars[i] +
                                                               (1 - self.opts['tau']) * target_critic_var)
            update_target_ops.append(update_target_critic_op)

        self.update_targets_op = tf.group(*update_target_ops)

        targets = tf.expand_dims(self.r_ph, 1) + \
                  tf.expand_dims(self.is_not_terminal_ph, 1) * self.opts['gamma'] * self.q_next_mu
        td_errors = targets - self.q_a_mu

        critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if 'b' not in var.name:
                critic_loss += self.opts['l2_reg_critic'] * 0.5 * tf.nn.l2_loss(var)

        critic_lr = self.opts['lr_critic'] * self.opts['lr_decay'] ** self.episodes
        self.critic_train_op = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss, var_list=critic_vars)

        actor_loss = -1 * tf.reduce_mean(self.q_sa_mu)
        actor_loss += gaussianNLL(self.a_ph, self.a_mu, self.a_sigma)

        for var in actor_vars:
            if 'b' not in var.name:
                actor_loss += self.opts['l2_reg_actor'] * 0.5 * tf.nn.l2_loss(var)

        actor_lr = self.opts['lr_actor'] * self.opts['lr_decay'] ** self.episodes
        self.actor_train_op = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=actor_vars)

    def initialize_u_with_gmm(self):
        gmm = BayesianGaussianMixture(n_components=self.opts['gmm_components'], covariance_type='full')
        gmm.fit(np.random.normal(size=(1000, self.opts['u_dim'])))  # Dummy data for fitting
        return gmm.sample(self.opts['u_sample_size'] * self.opts['batch_size'])[0].reshape(self.opts['u_sample_size'],
                                                                                           self.opts['batch_size'],
                                                                                           self.opts['u_dim'])

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

    def compute_z_init(self, x, a, r):
        z_init = self.sess.run(self.z_init, feed_dict={self.x: x, self.a: a, self.r: r})
        return z_init

    def compute_u_init(self, x, a, r):
        # Ensure inputs are NumPy arrays
        if isinstance(x, tf.Tensor):
            x = x.eval(session=self.sess)
        if isinstance(a, tf.Tensor):
            a = a.eval(session=self.sess)
        if isinstance(r, tf.Tensor):
            r = r.eval(session=self.sess)

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
        # r_next: batch_size x r_dim
        print(f"z: {type(z)}, a: {type(a)}, x: {type(x)}, u: {type(u)}")
        z_next_value, r_next_value, reward_samples = \
            self.sess.run([self.z_next, self.r_next, self.reward],
                        feed_dict={self.z: z, self.a: a, self.x: x, self.u: u})

        print(f"z_next_value: {type(z_next_value)}, r_next_value: {type(r_next_value)}, reward_samples: {type(reward_samples)}")

        # Convert r_next_value to a Tensor if not already
        r_next_value = tf.convert_to_tensor(r_next_value, dtype=tf.float32)

        done = np.abs(r_next_value - self.opts['final_reward']) == 0
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        return z_next_value, r_next_value, done, reward_samples
    def actor_net(self, z, reuse, is_training, trainable):
        mu, sigma = ac_fc_net(self.opts, z, self.opts['policy_net_layers'],
                              self.opts['policy_net_outlayers'], 'policy_net',
                              reuse=reuse, is_training=is_training, trainable=trainable)
        mu = mu * 2
        return mu, sigma

    def critic_net(self, z, a, reuse, is_training, trainable):
        z_a = tf.concat([z, a], axis=1)
        mu, sigma = ac_fc_net(self.opts, z_a, self.opts['value_net_layers'],
                              self.opts['value_net_outlayers'], 'value_net',
                              reuse=reuse, is_training=is_training, trainable=trainable)
        return mu, sigma

    def choose_action(self, z, is_training):
        action = self.sess.run(self.a_samples, feed_dict={self.z_ph: z, self.is_training_ph: is_training})
        return action

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    def calculate_optimal_action_probability(self, z):
        action_mu, action_sigma = self.actor_net(z, reuse=True, is_training=False, trainable=False)
        action_distribution = tf.distributions.Normal(loc=action_mu, scale=action_sigma)
        action_prob = self.sess.run(action_distribution.prob(action_mu))
        return action_prob

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

            reward_data_path = os.path.join(plots_dir, 'ac_decon_reward_data.txt')
            with open(reward_data_path, 'a+') as f:

                if episode > self.opts['episode_start'] and episode % self.opts['save_every_episode'] == 0:
                    reward_filename = 'ac_decon_reward_plot_epoch_{:d}.png'.format(episode)
                    save_reward_plots(self.opts, reward_list, reward_filename)
                    self.saver.save(self.sess, os.path.join(checkpoints_dir, 'policy_decon'), global_step=total_episode)

                total_reward = 0
                steps_in_episode = 0

                tr_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
                tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
                tr_x_init = np.reshape(data.x_train[tr_batch_ids][:, tr_nstep_ids, :],
                                    [self.opts['batch_size'], self.opts['x_dim']])
                tr_a_init = np.reshape(data.a_train[tr_batch_ids][:, tr_nstep_ids, :],
                                    [self.opts['batch_size'], self.opts['a_dim']])
                tr_r_init = np.reshape(data.r_train[tr_batch_ids][:, tr_nstep_ids, :],
                                    [self.opts['batch_size'], self.opts['r_dim']])

                z = self.compute_z_init(tr_x_init, tr_a_init, tr_r_init)
                u_est = self.compute_u_init(tr_x_init, tr_a_init, tr_r_init)

                for step in range(self.opts['max_steps_in_episode']):
                    action = self.choose_action(z, False)

                    z_next, reward, done, reward_samples = self.step(z, action, tr_x_init, u_est)

                    # Ensure reward is a scalar, and done is converted properly
                    reward = reward.numpy() if isinstance(reward, tf.Tensor) else reward
                    done = done.numpy() if isinstance(done, tf.Tensor) else done

                    total_reward += reward

                    self.add_to_memory((np.reshape(z, self.opts['z_dim']),
                                        np.reshape(action, self.opts['a_dim']),
                                        reward,
                                        np.reshape(z_next, self.opts['z_dim']),
                                        0.0 if done else 1.0))

                    # Update parameters using a mini-batch of experience
                    if total_steps % self.opts['train_every'] == 0 and len(self.replay_memory) >= self.opts['mini_batch_size']:
                        mini_batch = self.sample_from_memory(self.opts['mini_batch_size'])

                        # Convert mini-batch elements to Tensors
                        z_batch = np.asarray([elem[0] for elem in mini_batch])
                        a_batch = np.asarray([elem[1] for elem in mini_batch])
                        r_batch = np.asarray([elem[2] for elem in mini_batch])
                        z_next_batch = np.asarray([elem[3] for elem in mini_batch])
                        is_not_terminal_batch = np.asarray([elem[4] for elem in mini_batch])

                        # Ensure all batches are Tensors
                        z_batch = tf.convert_to_tensor(z_batch, dtype=tf.float32)
                        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
                        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
                        z_next_batch = tf.convert_to_tensor(z_next_batch, dtype=tf.float32)
                        is_not_terminal_batch = tf.convert_to_tensor(is_not_terminal_batch, dtype=tf.float32)

                        _, _ = self.sess.run(
                            [self.critic_train_op, self.actor_train_op],
                            feed_dict={
                                self.z_ph: z_batch,
                                self.a_ph: a_batch,
                                self.r_ph: r_batch,
                                self.z_next_ph: z_next_batch,
                                self.is_not_terminal_ph: is_not_terminal_batch,
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

                f.write('{:f}\n'.format(np.mean(reward_que)))

        self.saver.save(self.sess, os.path.join(checkpoints_dir, 'policy_decon'), global_step=total_episode)
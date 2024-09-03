import tensorflow as tf
import numpy as np
import os
import logging
from collections import deque

class AC_Con(object):

    def __init__(self, sess, opts, model):
        self.sess = sess
        self.opts = opts
        self.model = model
        self.saver = None

        np.random.seed(0)

        # Define placeholders
        self.x = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['x_dim']])
        self.z = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['z_dim']])
        self.a = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['a_dim']])
        self.r = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['r_dim']])

        # Compute initial hidden state z_0 given x_0, a_0, r_0 using Model_Con
        _, z_mu_init, z_cov_init = self.model.q_z_g_z_x_a_r(self.x, self.a, self.r)
        self.z_init = z_mu_init

        # Define placeholders for training
        self.z_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.a_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']])
        self.r_ph = tf.placeholder(tf.float32, shape=[None])
        self.z_next_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None])

        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

        # Initialize episode counter
        self.episodes = tf.Variable(0.0, trainable=False)
        self.episode_increase_op = self.episodes.assign_add(1)

        # Replay memory
        self.replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])

        # Metric lists
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.q_value_list = []
        self.reward_list = []

        # Construct architecture for Actor-Critic
        self.construct_actor_critic_networks()

    def construct_actor_critic_networks(self):
        # Actor network
        with tf.variable_scope('actor_net'):
            self.a_mu, self.a_sigma = self.actor_net(self.z_ph, False, self.is_training_ph, True)
        # Re-parameterization trick
        eps = tf.random_normal((1, self.opts['a_dim']), 0., 1., dtype=tf.float32)
        self.a_samples = tf.clip_by_value(self.a_mu + tf.multiply(eps, tf.sqrt(1e-8 + self.a_sigma)),
                                          -self.opts['a_range'], self.opts['a_range'])

        # Target actor network
        with tf.variable_scope('target_actor_net', reuse=False):
            self.target_a_mu, self.target_a_sigma = self.actor_net(self.z_next_ph, False, self.is_training_ph, False)
            self.target_a_mu = tf.stop_gradient(self.target_a_mu)
            self.target_a_sigma = tf.stop_gradient(self.target_a_sigma)

        # Critic network
        with tf.variable_scope('critic_net'):
            self.q_a_mu, self.q_a_sigma = self.critic_net(self.z_ph, self.a_ph, False, self.is_training_ph, True)
            self.q_sa_mu, self.q_sa_sigma = self.critic_net(self.z_ph, self.a_mu, True, self.is_training_ph, True)

        # Target critic network
        with tf.variable_scope('target_critic_net', reuse=False):
            self.q_next_mu, self.q_next_sigma = \
                self.critic_net(self.z_next_ph, self.target_a_mu, False, self.is_training_ph, False)
            self.q_next_mu = tf.stop_gradient(self.q_next_mu)
            self.q_next_sigma = tf.stop_gradient(self.q_next_sigma)

        # Collect variables for each network
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_net')

        # Construct ops for updating parameters of target networks
        self.update_targets_op = self.construct_update_ops(actor_vars, target_actor_vars, critic_vars, target_critic_vars)

        # Compute loss functions
        self.construct_loss_functions(actor_vars, critic_vars)

    def construct_update_ops(self, actor_vars, target_actor_vars, critic_vars, target_critic_vars):
        update_target_ops = []
        for i, target_actor_var in enumerate(target_actor_vars):
            update_target_actor_op = target_actor_var.assign(self.opts['tau'] * actor_vars[i] +
                                                             (1 - self.opts['tau']) * target_actor_var)
            update_target_ops.append(update_target_actor_op)

        for i, target_critic_var in enumerate(target_critic_vars):
            update_target_critic_op = target_critic_var.assign(self.opts['tau'] * critic_vars[i] +
                                                               (1 - self.opts['tau']) * target_critic_var)
            update_target_ops.append(update_target_critic_op)

        return tf.group(*update_target_ops)

    def construct_loss_functions(self, actor_vars, critic_vars):
        # One-step temporal difference error
        targets = tf.expand_dims(self.r_ph, 1) + \
                  tf.expand_dims(self.is_not_terminal_ph, 1) * self.opts['gamma'] * self.q_next_mu
        td_errors = targets - self.q_a_mu

        # Critic loss function
        self.critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if 'b' not in var.name:
                self.critic_loss += self.opts['l2_reg_critic'] * 0.5 * tf.nn.l2_loss(var)

        critic_lr = self.opts['lr_critic'] * self.opts['lr_decay'] ** self.episodes
        self.critic_train_op = tf.train.AdamOptimizer(critic_lr).minimize(self.critic_loss, var_list=critic_vars)

        # Actor loss function
        self.actor_loss = -1 * tf.reduce_mean(self.q_sa_mu)
        self.actor_loss += self.gaussianNLL(self.a_ph, self.a_mu, self.a_sigma)

        for var in actor_vars:
            if 'b' not in var.name:
                self.actor_loss += self.opts['l2_reg_actor'] * 0.5 * tf.nn.l2_loss(var)

        actor_lr = self.opts['lr_actor'] * self.opts['lr_decay'] ** self.episodes
        self.actor_train_op = tf.train.AdamOptimizer(actor_lr).minimize(self.actor_loss, var_list=actor_vars)

    def actor_net(self, z, reuse, is_training, trainable):
        return self.model.fc_net(z, self.opts['policy_net_layers'], self.opts['policy_net_outlayers'], 'policy_net')

    def critic_net(self, z, a, reuse, is_training, trainable):
        z_a = tf.concat([z, a], axis=1)
        return self.model.fc_net(z_a, self.opts['value_net_layers'], self.opts['value_net_outlayers'], 'value_net')

    def gaussianNLL(self, labels, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.square(labels - mu) / (sigma + 1e-8) + tf.log(sigma + 1e-8))

    def choose_action(self, z, is_training):
        action = self.sess.run(self.a_samples, feed_dict={self.z_ph: z, self.is_training_ph: is_training})
        return action

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    def train(self, data):
        # Set up directories
        work_dir = self.opts.get('work_dir', './training_results')
        logs_dir = os.path.join(work_dir, 'logs')
        plots_dir = os.path.join(work_dir, 'plots')
        checkpoints_dir = os.path.join(work_dir, 'policy_checkpoints')
        
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Set up logging
        log_file = os.path.join(logs_dir, 'ac_con_training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info("Training started")

        total_steps = 0
        total_episode = 0

        reward_que = deque(maxlen=100)
        reward_list = []

        self.create_env()

        for episode in range(self.opts['episode_num']):
            total_reward = 0
            steps_in_episode = 0

            tr_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
            tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
            tr_x_init = np.reshape(data.x_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['x_dim']])
            tr_a_init = np.reshape(data.a_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['a_dim']])
            tr_r_init = np.reshape(data.r_train[tr_batch_ids][:, tr_nstep_ids, :], [self.opts['batch_size'], self.opts['r_dim']])

            z = self.sess.run(self.z_init, feed_dict={self.x: tr_x_init, self.a: tr_a_init, self.r: tr_r_init})

            for step in range(self.opts['max_steps_in_episode']):
                action = self.choose_action(z, False)
                z_next, reward, done, _ = self.step(z, action, tr_x_init)

                total_reward += reward

                self.add_to_memory((np.reshape(z, self.opts['z_dim']), np.reshape(action, self.opts['a_dim']),
                                    reward, np.reshape(z_next, self.opts['z_dim']), 0.0 if done else 1.0))

                if total_steps % self.opts['train_every'] == 0 and len(self.replay_memory) >= self.opts['mini_batch_size']:
                    mini_batch = self.sample_from_memory(self.opts['mini_batch_size'])
                    critic_loss, _, q_val, actor_loss, _ = self.sess.run(
                        [self.critic_loss, self.critic_train_op, self.q_sa_mu, self.actor_loss, self.actor_train_op],
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

                    # Log metrics
                    self.actor_loss_list.append(actor_loss)
                    self.critic_loss_list.append(critic_loss)
                    self.q_value_list.append(np.mean(q_val))

                z = z_next
                total_steps += 1
                steps_in_episode += 1

                if done:
                    _ = self.sess.run(self.episode_increase_op)
                    break

            total_episode += 1

            # Logging after every episode
            avg_actor_loss = np.mean(self.actor_loss_list)
            avg_critic_loss = np.mean(self.critic_loss_list)
            avg_q_value = np.mean(self.q_value_list)

            logging.info(f'Episode {episode}: Steps in Episode: {steps_in_episode}, '
                         f'Total Reward: {total_reward:.4f}, '
                         f'Avg Actor Loss: {avg_actor_loss:.4f}, '
                         f'Avg Critic Loss: {avg_critic_loss:.4f}, '
                         f'Avg Q Value: {avg_q_value:.4f}')

            reward_que.append(total_reward)
            reward_list.append(np.mean(reward_que))

            # Clear the metric lists after each episode
            self.actor_loss_list.clear()
            self.critic_loss_list.clear()
            self.q_value_list.clear()

        self.saver.save(self.sess, os.path.join(checkpoints_dir, 'policy_con'), global_step=total_episode)
        logging.info("Training completed.")

    def create_env(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        default_checkpoint_path = "./model_con"
        checkpoint_path = self.opts.get('model_checkpoint', default_checkpoint_path)

        if checkpoint_path and os.path.exists(checkpoint_path + ".index"):
            self.saver.restore(self.sess, checkpoint_path)
            print(f"Model restored from checkpoint: {checkpoint_path}")
        else:
            print(f"No valid checkpoint found at {checkpoint_path}. Initializing variables.")

    def step(self, z, action, x_init):
        # Replace this method with your environment step logic
        # This is a placeholder function
        # You would compute the next state, reward, and done flag based on the environment
        z_next = z  # This should be computed based on your model/environment
        reward = np.random.rand()  # Placeholder for actual reward computation
        done = False  # Placeholder for actual done condition
        reward_samples = None  # Placeholder if you need to return additional samples
        return z_next, reward, done, reward_samples

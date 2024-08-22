#########################
## Author: Chaochao Lu ##
#########################
import numpy as np
import tensorflow as tf
import os
import random
from collections import deque
from utils import *
import logging
class Vanilla_AC:

    def __init__(self, sess, opts):
        self.sess = sess
        self.opts = opts

        np.random.seed(self.opts['seed'])
        tf.set_random_seed(self.opts['seed'])

        self.state_ph = tf.placeholder(tf.float32, shape=[None, self.opts['x_dim']], name="state")
        self.action_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']], name="action")
        self.reward_ph = tf.placeholder(tf.float32, shape=[None], name="reward")
        self.next_state_ph = tf.placeholder(tf.float32, shape=[None, self.opts['x_dim']], name="next_state")
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None], name="is_not_terminal")
        self.is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

        with tf.variable_scope('actor_net'):
            self.actor_mu, self.actor_sigma = self.build_actor_net(self.state_ph, trainable=True)
        with tf.variable_scope('actor_net', reuse=True):
            self.sampled_action = self.actor_mu + tf.multiply(tf.random_normal(tf.shape(self.actor_sigma)), self.actor_sigma)

        with tf.variable_scope('critic_net'):
            self.q_value = self.build_critic_net(self.state_ph, self.action_ph, trainable=True)

        with tf.variable_scope('target_actor_net'):
            self.target_actor_mu, self.target_actor_sigma = self.build_actor_net(self.next_state_ph, trainable=False)
        with tf.variable_scope('target_critic_net'):
            self.target_q_value = self.build_critic_net(self.next_state_ph, self.target_actor_mu, trainable=False)

        self.actor_loss, self.actor_train_op = self.actor_loss_and_train_op()
        self.critic_loss, self.critic_train_op = self.critic_loss_and_train_op()
        self.update_targets_op = self.update_target_networks()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50)

    def build_actor_net(self, state, trainable):
        hidden1 = tf.layers.dense(state, 300, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, 300, activation=tf.nn.relu, trainable=trainable)
        mu = tf.layers.dense(hidden2, self.opts['a_dim'], activation=tf.nn.tanh, trainable=trainable)
        sigma = tf.layers.dense(hidden2, self.opts['a_dim'], activation=tf.nn.softplus, trainable=trainable)
        return mu, sigma

    def build_critic_net(self, state, action, trainable):
        state_action = tf.concat([state, action], axis=-1)
        hidden1 = tf.layers.dense(state_action, 300, activation=tf.nn.relu, trainable=trainable)
        hidden2 = tf.layers.dense(hidden1, 300, activation=tf.nn.relu, trainable=trainable)
        q_value = tf.layers.dense(hidden2, 1, activation=None, trainable=trainable)
        return q_value

    def actor_loss_and_train_op(self):
        with tf.variable_scope('critic_net', reuse=True):
            q_value = self.build_critic_net(self.state_ph, self.actor_mu, trainable=True)
        loss = -tf.reduce_mean(q_value)
        optimizer = tf.train.AdamOptimizer(self.opts['lr_actor'])
        train_op = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net'))
        return loss, train_op

    def critic_loss_and_train_op(self):
        target_q = self.reward_ph + self.opts['gamma'] * self.is_not_terminal_ph * tf.squeeze(self.target_q_value)
        loss = tf.reduce_mean(tf.square(target_q - tf.squeeze(self.q_value)))
        optimizer = tf.train.AdamOptimizer(self.opts['lr_critic'])
        train_op = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net'))
        return loss, train_op

    def update_target_networks(self):
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        target_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        target_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_net')

        update_ops = []
        for src, dest in zip(actor_params, target_actor_params):
            update_ops.append(tf.assign(dest, self.opts['tau'] * src + (1.0 - self.opts['tau']) * dest))
        for src, dest in zip(critic_params, target_critic_params):
            update_ops.append(tf.assign(dest, self.opts['tau'] * src + (1.0 - self.opts['tau']) * dest))

        return tf.group(*update_ops)

    def choose_action(self, state, is_training):
        return self.sess.run(self.sampled_action, feed_dict={self.state_ph: state, self.is_training_ph: is_training})

    def train(self, data):
        replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])
        total_steps = 0
        reward_list = []

        log_dir = os.path.join(self.opts['work_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)
        logger = logging.getLogger()

        for episode in range(self.opts['episode_num']):
            tr_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
            tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
            state = data.x_train[tr_batch_ids][:, tr_nstep_ids, :].reshape(self.opts['batch_size'], self.opts['x_dim'])
            episode_reward = 0

            for step in range(self.opts['max_steps_in_episode']):
                action = self.choose_action(state, True)
                tr_nstep_ids_next = np.random.choice(self.opts['nsteps'], 1)
                next_state = data.x_train[np.random.choice(data.train_num, self.opts['batch_size'], replace=False)][:, tr_nstep_ids_next, :]
                next_state = next_state.reshape(self.opts['batch_size'], self.opts['x_dim'])
                reward = np.random.rand(self.opts['batch_size'])
                done = np.random.choice([0, 1], size=(self.opts['batch_size'],))

                replay_memory.append((state, action, reward, next_state, 1.0 - done))
                episode_reward += np.mean(reward)

                if len(replay_memory) >= self.opts['mini_batch_size']:
                    minibatch = random.sample(replay_memory, self.opts['mini_batch_size'])
                    states, actions, rewards, next_states, not_terminals = zip(*minibatch)
                    self.sess.run([self.critic_train_op, self.actor_train_op], feed_dict={
                        self.state_ph: np.array(states).reshape(-1, self.opts['x_dim']),
                        self.action_ph: np.array(actions).reshape(-1, self.opts['a_dim']),
                        self.reward_ph: np.array(rewards).reshape(-1),
                        self.next_state_ph: np.array(next_states).reshape(-1, self.opts['x_dim']),
                        self.is_not_terminal_ph: np.array(not_terminals).reshape(-1),
                        self.is_training_ph: True
                    })
                    self.sess.run(self.update_targets_op)

                state = next_state
                if done.any():
                    break

            reward_list.append(episode_reward)
            logger.info(f'Episode: {episode}, Reward: {episode_reward}')

            if (episode + 1) % self.opts['save_every_episode'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'policy_checkpoints', 'policy_vanilla'), global_step=episode)
                logger.info(f"Model checkpoint saved at episode {episode}")

        return reward_list

def main():
    import configs
    from data_handler import DataHandler

    opts = configs.model_config

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    print('starting processing data ...')
    data = DataHandler(opts)

    print('starting training policy using Vanilla_AC ...')
    ac = Vanilla_AC(sess, opts)
    reward_list = ac.train(data)
    print('Training completed.')

if __name__ == "__main__":
    main()

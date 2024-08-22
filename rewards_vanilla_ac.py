import numpy as np
import tensorflow as tf
import logging
import os
from collections import deque
import random
class ActorCritic:
    def __init__(self, opts):
        self.opts = opts
        self._build_networks()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50)

    def _build_networks(self):
        self.state_ph = tf.placeholder(tf.float32, [None, self.opts['x_dim']], name='state')
        self.action_ph = tf.placeholder(tf.int32, [None], name='action')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='reward')
        self.next_state_ph = tf.placeholder(tf.float32, [None, self.opts['x_dim']], name='next_state')
        self.is_not_terminal_ph = tf.placeholder(tf.float32, [None], name='is_not_terminal')
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')

        # Actor network
        with tf.variable_scope('actor'):
            actor_hidden = tf.layers.dense(self.state_ph, 128, activation=tf.nn.relu)
            self.sampled_action = tf.layers.dense(actor_hidden, self.opts['a_dim'], activation=None)

        # Critic network
        with tf.variable_scope('critic'):
            critic_hidden = tf.layers.dense(self.state_ph, 128, activation=tf.nn.relu)
            self.state_value = tf.layers.dense(critic_hidden, 1, activation=None)

        # Critic training
        with tf.variable_scope('target_critic'):
            target_critic_hidden = tf.layers.dense(self.next_state_ph, 128, activation=tf.nn.relu)
            self.target_state_value = tf.stop_gradient(tf.layers.dense(target_critic_hidden, 1, activation=None))

        self.critic_loss = tf.reduce_mean(tf.square(self.reward_ph + self.opts['gamma'] * self.is_not_terminal_ph * tf.squeeze(self.target_state_value) - tf.squeeze(self.state_value)))
        self.critic_train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(self.critic_loss)

        # Actor training
        action_mask = tf.one_hot(self.action_ph, self.opts['a_dim'])
        self.log_prob = tf.reduce_sum(action_mask * tf.nn.log_softmax(self.sampled_action), axis=1)
        self.advantage = self.reward_ph + self.opts['gamma'] * self.is_not_terminal_ph * tf.squeeze(self.target_state_value) - tf.squeeze(self.state_value)
        self.actor_loss = -tf.reduce_mean(self.log_prob * self.advantage)
        self.actor_train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(self.actor_loss)

        # Target network update
        self.update_targets_op = [tf.assign(t, e) for t, e in zip(tf.trainable_variables('target_critic'), tf.trainable_variables('critic'))]

    def choose_action(self, state, is_training):
        action = self.sess.run(self.sampled_action, feed_dict={self.state_ph: state, self.is_training_ph: is_training})
        return np.argmax(action, axis=1)

    def train(self, data):
        replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])
        total_steps = 0
        reward_list = []

        for episode in range(self.opts['episode_num']):
            tr_batch_ids = np.random.choice(data['train_num'], self.opts['batch_size'], replace=False)
            tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
            state = data['x_train'][tr_batch_ids][:, tr_nstep_ids, :].reshape(self.opts['batch_size'], self.opts['x_dim'])
            episode_reward = 0

            for step in range(self.opts['max_steps_in_episode']):
                action = self.choose_action(state, True)
                tr_nstep_ids_next = np.random.choice(self.opts['nsteps'], 1)
                next_state = data['x_train'][np.random.choice(data['train_num'], self.opts['batch_size'], replace=False)][:, tr_nstep_ids_next, :]
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
                        self.action_ph: np.array(actions).reshape(-1),
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

        return reward_list

def main():
    # Configure logging
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
    global logger
    logger = logging.getLogger()

    # Hyperparameters
    opts = {
        'x_dim': 784,
        'a_dim': 2,
        'gamma': 0.99,
        'lr': 0.001,
        'batch_size': 32,
        'mini_batch_size': 32,
        'replay_memory_capacity': 10000,
        'episode_num': 200,
        'max_steps_in_episode': 200,
        'save_every_episode': 50,
        'work_dir': './training_results',
        'nsteps': 5,
    }

    # Load data
    data = np.load('mnist_training_data.npz')
    data = {
        'x_train': data['x_train'],
        'a_train': data['a_train'],
        'r_train': data['r_train'],
        'mask_train': data['mask_train'],
        'rich_train': data['rich_train'],
        'train_num': len(data['x_train']),
    }

    ac = ActorCritic(opts)
    rewards = ac.train(data)

    # Save rewards
    reward_file = os.path.join(opts['work_dir'], 'rewards_vanilla.txt')
    with open(reward_file, 'w') as f:
        for reward in rewards:
            f.write(f"{reward}\n")

if __name__ == "__main__":
    main()

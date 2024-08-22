import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
import networkx as nx
import os
import logging
import matplotlib.pyplot as plt

class Model_Decon(object):

    def __init__(self, sess, opts):
        self.sess = sess
        self.opts = opts
        self.u = None
        self.u_x_list = []
        self.u_a_list = []
        self.u_r_list = []
        self.saver = None

    ########################################################################################################################
    ############################################# create generative model ##################################################
    ########################################################################################################################

    def p_z(self):
        mu_z = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        cov_z = tf.ones([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        eps = tf.random_normal((self.opts['batch_size'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z = mu_z + tf.multiply(eps, tf.sqrt(1e-8 + cov_z))
        return z

    def p_x_g_z_u(self, z, u):
        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = self.fc_net(z, self.opts['pxgz_net_layers'], self.opts['pxgz_net_outlayers'], 'pxgz_net')
        u_fea = self.fc_net(u, self.opts['pxgu_net_layers'], self.opts['pxgu_net_outlayers'], 'pxgu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        if self.opts['is_conv']:
            z_fea = self.fc_net(zu_fea, self.opts['pxgzu_prenet_layers'],
                                self.opts['pxgzu_prenet_outlayers'], 'pxgz_prenet')
            z_fea = tf.reshape(z_fea, z_fea.get_shape().as_list()[:-1] + [4, 4, 32])

            mu, sigma = self.decoder(z_fea, self.opts['pxgzu_in_shape'],
                                     self.opts['pxgzu_out_shape'], 'pxgzu_conv_net')

            mu = tf.reshape(mu, mu.get_shape().as_list()[:-3] + [-1])
            sigma = tf.reshape(sigma, sigma.get_shape().as_list()[:-3] + [-1])

        else:
            mu, sigma = self.fc_net(zu_fea, self.opts['pxgzu_net_layers'],
                                    self.opts['pxgzu_net_outlayers'], 'pxgzu_net')

        return mu, sigma

    def p_a_g_z_u(self, z, u):
        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = self.fc_net(z, self.opts['pagz_net_layers'], self.opts['pagz_net_outlayers'], 'pagz_net')
        u_fea = self.fc_net(u, self.opts['pagu_net_layers'], self.opts['pagu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        mu, sigma = self.fc_net(zu_fea, self.opts['pagzu_net_layers'],
                                self.opts['pagzu_net_outlayers'], 'pagzu_net')
        mu = mu * self.opts['a_range']

        return mu, sigma

    def p_r_g_z_a_u(self, z, a, u):
        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = self.fc_net(z, self.opts['prgz_net_layers'], self.opts['prgz_net_outlayers'], 'prgz_net')
        a_fea = self.fc_net(a, self.opts['prga_net_layers'], self.opts['prga_net_outlayers'], 'prga_net')
        u_fea = self.fc_net(u, self.opts['prgu_net_layers'], self.opts['prgu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 2)
        else:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 1)

        mu, sigma = self.fc_net(zau_fea, self.opts['prgzau_net_layers'],
                                self.opts['prgzau_net_outlayers'], 'prgzau_net')

        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        z_fea = self.fc_net(z, self.opts['pzgz_net_layers'], self.opts['pzgz_net_outlayers'], 'pzgz_net')
        a_fea = self.fc_net(a, self.opts['pzga_net_layers'], self.opts['pzga_net_outlayers'], 'pzga_net')

        if len(z.get_shape().as_list()) > 2:
            az_fea = tf.concat([z_fea, a_fea], 2)
        else:
            az_fea = tf.concat([z_fea, a_fea], 1)

        h_az_fea = self.fc_net(az_fea, self.opts['pzgza_net_layers'],
                               self.opts['pzgza_net_outlayers'], 'pzgza_net')
        h_mu = self.fc_net(h_az_fea, self.opts['pzgza_mu_net_layers'],
                           self.opts['pzgza_mu_net_outlayers'], 'pzgza_mu_net')

        if self.opts['gated']:
            hg_az_fea = self.fc_net(az_fea, self.opts['pzgza_pregate_net_layers'],
                                    self.opts['pzgza_pregate_net_outlayers'], 'pzgza_pregate_net')
            gate = self.fc_net(hg_az_fea, self.opts['pzgza_gate_net_layers'],
                               self.opts['pzgza_gate_net_outlayers'], 'pzgza_gate_net')
            mu = gate * h_mu + (1 - gate) * self.fc_net(az_fea, self.opts['pzgza_gate_mu_net_layers'],
                                                        self.opts['pzgza_gate_mu_net_outlayers'], 'pzgza_gate_mu_net')
        else:
            mu = h_mu

        sigma = self.fc_net(h_az_fea, self.opts['pzgza_sigma_net_layers'],
                            self.opts['pzgza_sigma_net_outlayers'], 'pzgza_sigma_net')

        return mu, sigma

    ########################################################################################################################
    ############################################# LSTM and Encoder Functions ###################################################
    ########################################################################################################################

    def lstm_cell(self, prev, current):
        h_prev = prev[0]
        c_prev = prev[1]
        x_current = current[0]
        mask = current[1]

        with tf.variable_scope('lstm_cell_'+self.fb, reuse=tf.AUTO_REUSE):
            w_i = tf.get_variable('w_i', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable('b_i', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_f = tf.get_variable('w_f', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable('b_f', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_o = tf.get_variable('w_o', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('b_o', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_c = tf.get_variable('w_c', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable('b_c', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))

        concat_x_h = tf.concat([x_current, h_prev], 1)
        i = tf.sigmoid(tf.matmul(concat_x_h, w_i) + b_i)
        f = tf.sigmoid(tf.matmul(concat_x_h, w_f) + b_f)
        o = tf.sigmoid(tf.matmul(concat_x_h, w_o) + b_o)
        c = tf.tanh(tf.matmul(concat_x_h, w_c) + b_c)

        c_new = tf.multiply(f, c_prev) + tf.multiply(i, c)
        c = tf.multiply(c_new, mask) + tf.multiply(1 - mask, c_prev)
        h_new = tf.multiply(o, tf.tanh(c))
        h = tf.multiply(h_new, mask) + tf.multiply(1 - mask, h_prev)

        return h, c

    def lstm_net(self, lstm_input, suffix, mask=None):
        lstm_input = tf.transpose(lstm_input, [1, 0, 2])
        lstm_input_shape = lstm_input.get_shape().as_list()

        lstm_embed = self.fc_net(lstm_input, self.opts['lstm_net_layers'],
                                 self.opts['lstm_net_outlayers'], 'lstm_net')

        if mask is None:
            mask = tf.ones([lstm_input_shape[0], self.opts['batch_size'], self.opts['lstm_dim']])
        else:
            mask = tf.tile(mask, [1, 1, self.opts['lstm_dim']])
            mask = tf.transpose(mask, [1, 0, 2])

        if suffix == 'R' or suffix == 'UR':
            self.fb = 'backward' + '_' + suffix

            lstm_embed = tf.reverse(lstm_embed, [0])
            mask = tf.reverse(mask, [0])

            lstm_embed = lstm_embed[:, tf.newaxis]
            mask = mask[:, tf.newaxis]
            lm_concat = tf.concat([lstm_embed, mask], 1)
            lm_split = tf.split(lm_concat, lstm_input_shape[0], 0)
            lm_list = []
            for i in range(lstm_input_shape[0]):
                lm_list.append(tf.reshape(lm_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

            elements = tf.convert_to_tensor(lm_list)

            output_backward = tf.scan(
                self.lstm_cell,
                elements,
                initializer=(tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]),
                             tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]))
            )

            lstm_output = output_backward[0]
            lstm_output = tf.reverse(lstm_output, [0])
        else:
            self.fb = 'forward' + '_' + suffix

            lstm_embed = lstm_embed[:, tf.newaxis]
            mask = mask[:, tf.newaxis]
            lm_concat = tf.concat([lstm_embed, mask], 1)
            lm_split = tf.split(lm_concat, lstm_input_shape[0], 0)
            lm_list = []
            for i in range(lstm_input_shape[0]):
                lm_list.append(tf.reshape(lm_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

            elements = tf.convert_to_tensor(lm_list)

            output_forward = tf.scan(
                self.lstm_cell,
                elements,
                initializer=(tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]),
                             tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]))
            )

            lstm_output = output_forward[0]

        return self.lstm_dropout(lstm_output, self.opts['lstm_dropout_prob'])

    def encoder(self, x, channels, output_channels, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = tf.layers.conv2d(x, channels, 4, strides=2, padding='same', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, output_channels, 4, strides=2, padding='same', activation=tf.nn.relu)
            flat = tf.layers.flatten(conv2)
            return flat

    def decoder(self, z, output_shape, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            fc = tf.layers.dense(z, units=np.prod(output_shape) // 4, activation=tf.nn.relu)
            fc = tf.reshape(fc, [-1] + output_shape[:-1] + [output_shape[-1] // 4])
            deconv1 = tf.layers.conv2d_transpose(fc, output_shape[-1] // 2, 4, strides=2, padding='same', activation=tf.nn.relu)
            deconv2 = tf.layers.conv2d_transpose(deconv1, output_shape[-1], 4, strides=2, padding='same', activation=tf.nn.sigmoid)
            return deconv2

    def lstm_dropout(self, lstm_output, dropout_prob):
        return tf.nn.dropout(lstm_output, dropout_prob)

    def fc_net(self, inputs, layers, out_layers, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = inputs
            for i, layer in enumerate(layers):
                net = tf.layers.dense(net, units=layer, activation=tf.nn.relu, name=f'fc_{i}')
            net = tf.layers.dense(net, units=out_layers, activation=None, name='fc_out')
            return net

    ########################################################################################################################
    ############################################# Model Inference and Training ################################################
    ########################################################################################################################

    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        x_seq_dim = 3
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)
            x_seq_dim = 2

        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.get_shape().as_list()[:-1] + [28, 28, 1])
            x_encoded = self.encoder(x_reshape, self.opts['qzgx_in_channels'],
                                     self.opts['qzgx_out_channel'], 'qzgx_conv_net')
            x_fea = self.fc_net(x_encoded, self.opts['qzgx_encoded_net_layers'],
                                self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net')
        else:
            x_fea = self.fc_net(x_seq, self.opts['qzgx_net_layers'],
                                self.opts['qzgx_net_outlayers'], 'qzgx_net')

        a_fea = self.fc_net(a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = self.fc_net(r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = self.fc_net(concat_xar, self.opts['qzgxar_net_layers'],
                              self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        h_r = self.lstm_net(xar_fea, 'R', mask)

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'L', mask)
            h = (h_r + h_l) / 2.
        else:
            h = h_r

        z_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        mu_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']])
        cov_0 = tf.ones([self.opts['batch_size'], self.opts['z_dim']])

        h = h[:, tf.newaxis]
        a_fea = self.fc_net(a_fea, self.opts['qagh_net_layers'], self.opts['qagh_net_outlayers'], 'qagh_net')
        a_fea = tf.transpose(a_fea, [1, 0, 2])
        a_fea = tf.concat([tf.ones([1, self.opts['batch_size'], tf.shape(a_fea)[2]]), a_fea[:-1, :, :]], 0)
        a_fea = a_fea[:, tf.newaxis]

        ha_concat = tf.concat([h, a_fea], 1)
        ha_split = tf.split(ha_concat, x_seq.get_shape().as_list()[1], 0)
        ha_list = []
        for i in range(x_seq.get_shape().as_list()[1]):
            ha_list.append(tf.reshape(ha_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

        elements = tf.convert_to_tensor(ha_list)

        output_q = tf.scan(
            self.st_approx,
            elements,
            initializer=(z_0, mu_0, cov_0)
        )

        z = tf.transpose(output_q[0], [1, 0, 2])
        mu = tf.transpose(output_q[1], [1, 0, 2])
        cov = tf.transpose(output_q[2], [1, 0, 2])

        if x_seq_dim == 2:
            z = tf.squeeze(z, [1])
            mu = tf.squeeze(mu, [1])
            cov = tf.squeeze(cov, [1])

        return z, mu, cov

    def q_u_g_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        x_seq_dim = 3
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)
            x_seq_dim = 2

        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.get_shape().as_list()[:-1] + [28, 28, 1])
            x_encoded = self.encoder(x_reshape, self.opts['qzgx_in_channels'],
                                     self.opts['qzgx_out_channel'], 'qzgx_conv_net', reuse=True)
            x_fea = self.fc_net(x_encoded, self.opts['qzgx_encoded_net_layers'],
                                self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net', reuse=True)
        else:
            x_fea = self.fc_net(x_seq, self.opts['qugx_net_layers'],
                                self.opts['qugx_net_outlayers'], 'qugx_net')

        a_fea = self.fc_net(a_seq, self.opts['qzga_net_layers'],
                            self.opts['qzga_net_outlayers'], 'qzga_net', reuse=True)
        r_fea = self.fc_net(r_seq, self.opts['qzgr_net_layers'],
                            self.opts['qzgr_net_outlayers'], 'qzgr_net', reuse=True)

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = self.fc_net(concat_xar, self.opts['qzgxar_net_layers'],
                              self.opts['qzgxar_net_outlayers'], 'qzgxar_net', reuse=True)

        h_r = self.lstm_net(xar_fea, 'UR', mask)
        h_r = tf.reverse(h_r, [0])

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'UL', mask)
            h_l = tf.reverse(h_l, [0])

            h = (h_r[0] + h_l[0]) / 2.
        else:
            h = h_r[0]

        h_trans = tf.reshape(h, [self.opts['batch_size'], -1])

        # Automatically determine clusters using GMM
        gmm = GaussianMixture(n_components=self.opts['gmm_components'], covariance_type='diag')
        gmm.fit(h_trans)
        u_mu = gmm.means_
        u_cov = gmm.covariances_

        return u_mu, u_cov

    def neg_elbo(self, x_seq, a_seq, r_seq, u_seq, anneal=1, mask=None):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)

        eps = tf.random_normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)

        mu_prior = tf.concat([tf.zeros([self.opts['batch_size'], 1, self.opts['z_dim']]), mu_p[:, :-1, :]], 1)
        cov_prior = tf.concat([tf.ones([self.opts['batch_size'], 1, self.opts['z_dim']]), cov_p[:, :-1, :]], 1)

        kl_divergence = self.gaussian_kl(mu_prior, cov_prior, mu_q, cov_q, mask)

        u_mu, u_cov = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)

        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, u_mu)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, u_mu)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, u_mu)

        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = self.gaussian_nll(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = self.gaussian_nll(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = self.gaussian_nll(r_seq, mu_prgza, cov_prgza, mask)

        nll_qagx = self.gaussian_nll(a_seq, mu_qagx, cov_qagx, mask)
        nll_qrgxa = self.gaussian_nll(r_seq, mu_qrgxa, cov_qrgxa, mask)

        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + nll_qagx + nll_qrgxa

        return nll, kl_divergence, u_mu

    ########################################################################################################################
    ############################################# SCM Functions ###########################################################
    ########################################################################################################################

    def build_scm_graph(self):
        edges = [
            ('z', 'x'),
            ('u', 'x'),
            ('z', 'a'),
            ('u', 'a'),
            ('z', 'r'),
            ('a', 'r'),
            ('u', 'r')
        ]
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    def visualize_scm(self, G):
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
        plt.title('Structural Causal Model (SCM)')
        plt.show()

    def identify_backdoor_paths(self, G, treatment, outcome):
        all_paths = list(nx.all_simple_paths(G, source=treatment, target=outcome))
        conditioning_set = []

        for node in G.nodes():
            if node == treatment or node == outcome:
                continue
            blocked_paths = all(nx.has_path(G, source=node, target=outcome) for path in all_paths)
            if blocked_paths:
                conditioning_set.append(node)

        return conditioning_set

    ########################################################################################################################
    ############################################# Train the Model ##########################################################
    ########################################################################################################################

    def train_model(self, data):
        # Ensure the directory for the log file exists
        log_dir = os.path.join(self.opts['work_dir'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, 'mnist.log')

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
        logger = logging.getLogger()

        batch_num = np.floor(data.train_num / self.opts['batch_size']).astype(int)
        counter = self.opts['counter_start']

        x_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['x_dim']])
        a_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['a_dim']])
        r_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['r_dim']])
        u_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['u_dim']])
        mask = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['mask_dim']])

        loss_gt = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], None])
        loss_recons = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], None])

        re_loss = self.recons_loss(self.opts['recons_cost'], loss_gt, loss_recons)
        nll, kl_dist, kl_dist_u = self.neg_elbo(x_seq, a_seq, r_seq, u_seq, anneal=self.opts['anneal'], mask=mask)
        x_recons, a_recons, r_recons = self.recons_xar_seq_g_xar_seq(x_seq, a_seq, r_seq, mask)

        train_sample_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)
        train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(nll)

        logger.info('Starting initializing variables ...')

        if self.opts['is_restored']:
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(all_vars, max_to_keep=50)
            self.saver.restore(self.sess, self.opts['model_checkpoint'])
        else:
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=50)

        logger.info('Starting epoch ...')

        for epoch in range(self.opts['epoch_start'], self.opts['epoch_start'] + self.opts['epoch_num']):

            if epoch > self.opts['epoch_start'] and epoch % self.opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_decon'),
                                global_step=counter)

            ids_perm = np.random.permutation(data.train_num)

            for itr in range(batch_num):
                start_time = time.time()
                batch_ids = ids_perm[self.opts['batch_size'] * itr:self.opts['batch_size'] * (itr + 1)]

                _, nll_tr = self.sess.run([train_op, nll], feed_dict={x_seq: data.x_train[batch_ids],
                                                                      a_seq: data.a_train[batch_ids],
                                                                      r_seq: data.r_train[batch_ids],
                                                                      u_seq: data.rich_train[batch_ids],
                                                                      mask: data.mask_train[batch_ids]})

                x_recons_tr, a_recons_tr, r_recons_tr = self.sess.run([x_recons, a_recons, r_recons],
                                                                      feed_dict={x_seq: data.x_train[train_sample_batch_ids],
                                                                                 a_seq: data.a_train[train_sample_batch_ids],
                                                                                 r_seq: data.r_train[train_sample_batch_ids],
                                                                                 u_seq: data.rich_train[train_sample_batch_ids],
                                                                                 mask: data.mask_train[train_sample_batch_ids]})

                x_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.x_train[train_sample_batch_ids],
                                                              loss_recons: x_recons_tr})
                a_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.a_train[train_sample_batch_ids],
                                                              loss_recons: a_recons_tr})
                r_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.r_train[train_sample_batch_ids],
                                                              loss_recons: r_recons_tr})

                elapsed_time = time.time() - start_time

                logger.info(
                    'epoch: {:d}, itr: {:d}, nll_tr: {:f}, x_tr_loss: {:f}, a_tr_loss: {:f}, r_tr_loss: {:f}, elapsed_time: {:f}'.format(
                        epoch, itr, nll_tr, x_tr_loss, a_tr_loss, r_tr_loss, elapsed_time))

        self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_decon'),
                        global_step=counter)

        # SCM Visualization and Analysis
        scm_graph = self.build_scm_graph()
        self.visualize_scm(scm_graph)

        conditioning_set = self.identify_backdoor_paths(scm_graph, treatment='a', outcome='r')
        print("Conditioning set for backdoor criterion:", conditioning_set)
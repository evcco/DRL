#########################
## Author: Aymen Nasri ##
#########################
import numpy as np
import tensorflow as tf
import os

from utils import *

class Model_Con(object):

    def __init__(self, sess, opts):
        self.sess = sess
        self.opts = opts
        np.random.seed(self.opts['seed'])

        self.confounder_u = None

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

        z_fea = fc_net(self.opts, z, self.opts['pxgz_net_layers'], self.opts['pxgz_net_outlayers'], 'pxgz_net')
        u_fea = fc_net(self.opts, u, self.opts['pxgu_net_layers'], self.opts['pxgu_net_outlayers'], 'pxgu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zu_fea, self.opts['pxgzu_net_layers'],
                           self.opts['pxgzu_net_outlayers'], 'pxgzu_net')

        return mu, sigma

    def p_a_g_z_u(self, z, u):
        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['pagz_net_layers'], self.opts['pagz_net_outlayers'], 'pagz_net')
        u_fea = fc_net(self.opts, u, self.opts['pagu_net_layers'], self.opts['pagu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zu_fea, self.opts['pagzu_net_layers'],
                           self.opts['pagzu_net_outlayers'], 'pagzu_net')
        mu = mu * self.opts['a_range']

        return mu, sigma

    def p_r_g_z_a_u(self, z, a, u):
        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['prgz_net_layers'], self.opts['prgz_net_outlayers'], 'prgz_net')
        a_fea = fc_net(self.opts, a, self.opts['prga_net_layers'], self.opts['prga_net_outlayers'], 'prga_net')
        u_fea = fc_net(self.opts, u, self.opts['prgu_net_layers'], self.opts['prgu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 2)
        else:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zau_fea, self.opts['prgzau_net_layers'],
                           self.opts['prgzau_net_outlayers'], 'prgzau_net')

        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        z_fea = fc_net(self.opts, z, self.opts['pzgz_net_layers'], self.opts['pzgz_net_outlayers'], 'pzgz_net')
        a_fea = fc_net(self.opts, a, self.opts['pzga_net_layers'], self.opts['pzga_net_outlayers'], 'pzga_net')

        if len(z.get_shape().as_list()) > 2:
            az_fea = tf.concat([z_fea, a_fea], 2)
        else:
            az_fea = tf.concat([z_fea, a_fea], 1)

        h_az_fea = fc_net(self.opts, az_fea, self.opts['pzgza_net_layers'],
                          self.opts['pzgza_net_outlayers'], 'pzgza_net')
        mu = fc_net(self.opts, h_az_fea, self.opts['pzgza_mu_net_layers'],
                      self.opts['pzgza_mu_net_outlayers'], 'pzgza_mu_net')

        sigma = fc_net(self.opts, h_az_fea, self.opts['pzgza_sigma_net_layers'],
                       self.opts['pzgza_sigma_net_outlayers'], 'pzgza_sigma_net')

        return mu, sigma

    ########################################################################################################################
    ############################################# create inference/recognition model #######################################
    ########################################################################################################################

    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        # Inference model for latent variable z given (x, a, r)
        
        # Ensure that the inputs are 3D by adding an extra dimension if they are 2D
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
        if len(a_seq.get_shape().as_list()) == 2:
            a_seq = tf.expand_dims(a_seq, 1)
        if len(r_seq.get_shape().as_list()) == 2:
            r_seq = tf.expand_dims(r_seq, 1)

        # Apply fully connected layers
        x_fea = fc_net(self.opts, x_seq, self.opts['qzgx_net_layers'],
                    self.opts['qzgx_net_outlayers'], 'qzgx_net')
        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        # Concatenate the features along the last dimension
        xar_fea = tf.concat([x_fea, a_fea, r_fea], axis=-1)
        
        # Compute the hidden representation
        h_r = fc_net(self.opts, xar_fea, self.opts['qzgxar_net_layers'], self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        # Compute the mean and covariance
        mu = fc_net(self.opts, h_r, self.opts['qzgz_net_layers'], self.opts['qzgz_mu_outlayers'], 'qzgz_mu_net')
        cov = fc_net(self.opts, h_r, self.opts['qzgz_net_layers'], self.opts['qzgz_sigma_outlayers'], 'qzgz_sigma_net')

        return mu, cov


    def q_u_g_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        # Inference model for the confounder u given (x, a, r)
        x_fea = fc_net(self.opts, x_seq, self.opts['qugx_net_layers'], self.opts['qugx_net_outlayers'], 'qugx_net')
        a_fea = fc_net(self.opts, a_seq, self.opts['quga_net_layers'], self.opts['quga_net_outlayers'], 'quga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qugr_net_layers'], self.opts['qugr_net_outlayers'], 'qugr_net')

        xar_fea = tf.concat([x_fea, a_fea, r_fea], 2)
        h_r = fc_net(self.opts, xar_fea, self.opts['qugxar_net_layers'], self.opts['qugxar_net_outlayers'], 'qugxar_net')

        mu = fc_net(self.opts, h_r, self.opts['qugz_net_layers'], self.opts['qugz_mu_outlayers'], 'qugz_mu_net')
        cov = fc_net(self.opts, h_r, self.opts['qugz_net_layers'], self.opts['qugz_sigma_outlayers'], 'qugz_sigma_net')

        return mu, cov

    ########################################################################################################################
    ############################################# create neg_elbo ##########################################################
    ########################################################################################################################

    def neg_elbo(self, x_seq, a_seq, r_seq, u_seq, anneal=1, mask=None):
        # z_q, mu_q, cov_q: batch_size x nsteps x z_dim
        mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)

        # Sample z from the inference model
        eps = tf.random_normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']), 0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, u_seq)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, u_seq)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, u_seq)

        nll_pxgz = gaussianNLL(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = gaussianNLL(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = gaussianNLL(r_seq, mu_prgza, cov_prgza, mask)

        kl_divergence = gaussianKL(mu_q, cov_q, tf.zeros_like(mu_q), tf.ones_like(cov_q), mask)
        u_kl_divergence = gaussianKL(u_seq, tf.ones_like(u_seq), tf.zeros_like(u_seq), tf.ones_like(u_seq), mask)

        # Total loss is the negative ELBO
        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + u_kl_divergence
        return nll, kl_divergence

    ########################################################################################################################
    ############################################# training the model ########################################################
    ########################################################################################################################

    def train_model(self, data):
        batch_num = np.floor(data.train_num / self.opts['batch_size']).astype(int)
        counter = self.opts['counter_start']

        x_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['x_dim']])
        a_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['a_dim']])
        r_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['r_dim']])
        u_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['u_dim']])
        mask = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['mask_dim']])

        nll, kl_dist = self.neg_elbo(x_seq, a_seq, r_seq, u_seq, anneal=self.opts['anneal'], mask=mask)

        train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(nll)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50)

        for epoch in range(self.opts['epoch_num']):
            ids_perm = np.random.permutation(data.train_num)
            for itr in range(batch_num):
                batch_ids = ids_perm[self.opts['batch_size']*itr:self.opts['batch_size']*(itr+1)]
                _, nll_tr, kl_dist_tr = self.sess.run(
                    [train_op, nll, kl_dist],
                    feed_dict={
                        x_seq: data.x_train[batch_ids],
                        a_seq: data.a_train[batch_ids],
                        r_seq: data.r_train[batch_ids],
                        u_seq: data.rich_train[batch_ids],
                        mask: data.mask_train[batch_ids]
                    }
                )
                print(f'Epoch: {epoch}, Iteration: {itr}, NLL: {nll_tr}, KL Div: {kl_dist_tr}')

            if epoch % self.opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'confounded_checkpoints', 'model_confounded'), global_step=epoch)

        self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'confounded_checkpoints', 'model_confounded'), global_step=epoch)


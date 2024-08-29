import numpy as np
import tensorflow as tf
import os
import time
from utils import *
import logging
class Model_Con(object):

    def __init__(self, sess, opts):
        self.sess = sess
        self.opts = opts

        np.random.seed(self.opts['seed'])

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

    def p_x_g_z(self, z):
        z_fea = fc_net(self.opts, z, self.opts['pxgz_net_layers'], self.opts['pxgz_net_outlayers'], 'pxgz_net')

        if self.opts['is_conv']:
            z_fea = fc_net(self.opts, z_fea, self.opts['pxgzu_prenet_layers'],
                           self.opts['pxgzu_prenet_outlayers'], 'pxgz_prenet')
            z_fea = tf.reshape(z_fea, z_fea.get_shape().as_list()[:-1] + [4, 4, 32])

            mu, sigma = decoder(self.opts, z_fea, self.opts['pxgzu_in_shape'],
                                self.opts['pxgzu_out_shape'], 'pxgzu_conv_net')

            mu = tf.reshape(mu, mu.get_shape().as_list()[:-3] + [-1])
            sigma = tf.reshape(sigma, sigma.get_shape().as_list()[:-3] + [-1])

        else:
            mu, sigma = fc_net(self.opts, z_fea, self.opts['pxgzu_net_layers'],
                               self.opts['pxgzu_net_outlayers'], 'pxgzu_net')

        return mu, sigma

    def p_a_g_z(self, z):
        z_fea = fc_net(self.opts, z, self.opts['pagz_net_layers'], self.opts['pagz_net_outlayers'], 'pagz_net')
        mu, sigma = fc_net(self.opts, z_fea, self.opts['pagzu_net_layers'],
                           self.opts['pagzu_net_outlayers'], 'pagzu_net')
        mu = mu * self.opts['a_range']

        return mu, sigma

    def p_r_g_z_a(self, z, a):
        z_fea = fc_net(self.opts, z, self.opts['prgz_net_layers'], self.opts['prgz_net_outlayers'], 'prgz_net')
        a_fea = fc_net(self.opts, a, self.opts['prga_net_layers'], self.opts['prga_net_outlayers'], 'prga_net')

        zau_fea = tf.concat([z_fea, a_fea], 1)
        mu, sigma = fc_net(self.opts, zau_fea, self.opts['prgzau_net_layers'],
                           self.opts['prgzau_net_outlayers'], 'prgzau_net')

        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        z_fea = fc_net(self.opts, z, self.opts['pzgz_net_layers'], self.opts['pzgz_net_outlayers'], 'pzgz_net')
        a_fea = fc_net(self.opts, a, self.opts['pzga_net_layers'], self.opts['pzga_net_outlayers'], 'pzga_net')

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

    def q_z_g_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        x_fea = fc_net(self.opts, x_seq, self.opts['qzgx_net_layers'], self.opts['qzgx_net_outlayers'], 'qzgx_net')
        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'],
                         self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        h_r = self.lstm_net(xar_fea, 'R', mask)
        h = h_r

        z_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        mu_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']])
        cov_0 = tf.ones([self.opts['batch_size'], self.opts['z_dim']])

        h = h[:, tf.newaxis]
        a_fea = fc_net(self.opts, a_fea, self.opts['qagh_net_layers'], self.opts['qagh_net_outlayers'], 'qagh_net')
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

        return z, mu, cov

    def q_a_g_x(self, x):
        mu, sigma = fc_net(self.opts, x, self.opts['qagx_net_layers'],
                           self.opts['qagx_net_outlayers'], 'qagx_net')
        mu = mu * self.opts['a_range']
        return mu, sigma

    def q_r_g_x_a(self, x, a):
        x_fea = fc_net(self.opts, x, self.opts['qrgx_net_layers'], self.opts['qrgx_net_outlayers'], 'qrgx_net')
        a_fea = fc_net(self.opts, a, self.opts['qrga_net_layers'], self.opts['qrga_net_outlayers'], 'qrga_net')

        ax_fea = tf.concat([x_fea, a_fea], 1)

        mu, sigma = fc_net(self.opts, ax_fea, self.opts['qrgxa_net_layers'],
                           self.opts['qrgxa_net_outlayers'], 'qrgxa_net')
        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']

        return mu, sigma
    
    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        # x_seq should be: batch_size x nsteps x dim
        x_seq_dim = 3
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)
            x_seq_dim = 2

        x_fea = fc_net(self.opts, x_seq, self.opts['qzgx_net_layers'], self.opts['qzgx_net_outlayers'], 'qzgx_net')
        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'], self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        h_r = self.lstm_net(xar_fea, 'R', mask)
        h = h_r

        z_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        mu_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']])
        cov_0 = tf.ones([self.opts['batch_size'], self.opts['z_dim']])

        h = h[:, tf.newaxis]
        a_fea = fc_net(self.opts, a_fea, self.opts['qagh_net_layers'], self.opts['qagh_net_outlayers'], 'qagh_net')
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

        return z, mu, cov

########################################################################################################################
############################################# create neg_elbo ##########################################################
########################################################################################################################

    def neg_elbo(self, x_seq, a_seq, r_seq, mask=None):
        z_q, mu_q, cov_q = self.q_z_g_x_a_r(x_seq, a_seq, r_seq, mask)

        eps = tf.random_normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)

        mu_prior = tf.concat([tf.zeros([self.opts['batch_size'], 1, self.opts['z_dim']]), mu_p[:, :-1, :]], 1)
        cov_prior = tf.concat([tf.ones([self.opts['batch_size'], 1, self.opts['z_dim']]), cov_p[:, :-1, :]], 1)

        kl_divergence = gaussianKL(mu_prior, cov_prior, mu_q, cov_q, mask)

        mu_pxgz, cov_pxgz = self.p_x_g_z(z_q_samples)
        mu_pagz, cov_pagz = self.p_a_g_z(z_q_samples)
        mu_prgza, cov_prgza = self.p_r_g_z_a(z_q_samples, a_seq)

        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = gaussianNLL(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = gaussianNLL(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = gaussianNLL(r_seq, mu_prgza, cov_prgza, mask)

        nll_qagx = gaussianNLL(a_seq, mu_qagx, cov_qagx, mask)
        nll_qrgxa = gaussianNLL(r_seq, mu_qrgxa, cov_qrgxa, mask)

        nll = nll_pxgz + nll_pagz + nll_prgza + kl_divergence + nll_qagx + nll_qrgxa

        return nll, kl_divergence

    def gen_st_approx(self, prev, current):
        z_prev = prev[0]
        x_prev, _ = self.p_x_g_z(z_prev)
        a_prev = 2.*(2.*tf.random_uniform((self.opts['batch_size'], self.opts['a_dim']), 0., 1., dtype=tf.float32)-1)
        r_prev, _ = self.p_r_g_z_a(z_prev, a_prev)

        z_current_mu, _ = self.p_z_g_z_a(z_prev, a_prev)

        return z_current_mu, x_prev, a_prev, r_prev

    def gen_xar_seq_g_z(self, z_0):
        z_0_shape = z_0.get_shape().as_list()
        if len(z_0_shape) > 2:
            z_0 = tf.reshape(z_0, [z_0_shape[0], z_0_shape[2]])

        output_xar = tf.scan(
            self.gen_st_approx,
            tf.range(self.opts['nsteps']),
            initializer=(z_0, tf.zeros([self.opts['batch_size'], self.opts['x_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['a_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['r_dim']]))
        )

        return tf.transpose(output_xar[1], [1, 0, 2])

########################################################################################################################
############################################# train the model ##########################################################
########################################################################################################################

    def train_model(self, data):
        batch_num = np.floor(data.train_num / self.opts['batch_size']).astype(int)
        counter = self.opts['counter_start']

        train_nll = []
        train_kl = []
        train_x_loss = []
        train_a_loss = []
        train_r_loss = []

        validation_nll = []
        validation_kl = []
        validation_x_loss = []
        validation_a_loss = []
        validation_r_loss = []

        x_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['x_dim']])
        a_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['a_dim']])
        r_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['r_dim']])
        mask = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['mask_dim']])

        nll, kl_dist = self.neg_elbo(x_seq, a_seq, r_seq, mask=mask)
        x_recons, a_recons, r_recons = self.recons_xar_seq_g_xar_seq(x_seq, a_seq, r_seq, mask)

        x_0_sample = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], 1, self.opts['x_dim']])
        z_0_sample = self.gen_z_g_x(x_0_sample)
        x_seq_sample = self.gen_xar_seq_g_z(z_0_sample)

        train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(nll)

        print('starting initializing variables ...')

        if self.opts['is_restored']:
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(all_vars, max_to_keep=50)
            self.saver.restore(self.sess, self.opts['model_checkpoint'])
        else:
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=50)

        print('starting epoch ...')

        for epoch in range(self.opts['epoch_start'], self.opts['epoch_start']+self.opts['epoch_num']):

            if epoch > self.opts['epoch_start'] and epoch % self.opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_nodecon'),
                                global_step=counter)

            ids_perm = np.random.permutation(data.train_num)

            for itr in range(batch_num):
                start_time = time.time()

                batch_ids = ids_perm[self.opts['batch_size']*itr:self.opts['batch_size']*(itr+1)]

                _, nll_tr, kl_dist_tr = \
                    self.sess.run([train_op, nll, kl_dist],
                                  feed_dict={x_seq: data.x_train[batch_ids],
                                             a_seq: data.a_train[batch_ids],
                                             r_seq: data.r_train[batch_ids],
                                             mask: data.mask_train[batch_ids]})

                x_recons_tr, a_recons_tr, r_recons_tr = \
                    self.sess.run([x_recons, a_recons, r_recons],
                                  feed_dict={x_seq: data.x_train[batch_ids],
                                             a_seq: data.a_train[batch_ids],
                                             r_seq: data.r_train[batch_ids],
                                             mask: data.mask_train[batch_ids]})

                x_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.x_train[batch_ids],
                                                              loss_recons: x_recons_tr})
                a_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.a_train[batch_ids],
                                                              loss_recons: a_recons_tr})
                r_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.r_train[batch_ids],
                                                              loss_recons: r_recons_tr})

                train_nll.append(nll_tr)
                train_kl.append(kl_dist_tr)
                train_x_loss.append(x_tr_loss)
                train_a_loss.append(a_tr_loss)
                train_r_loss.append(r_tr_loss)

                ####################### validation ####################################################################

                nll_te, kl_dist_te, x_recons_te, a_recons_te, r_recons_te = \
                    self.sess.run([nll, kl_dist, x_recons, a_recons, r_recons],
                                  feed_dict={x_seq: data.x_validation[batch_ids],
                                             a_seq: data.a_validation[batch_ids],
                                             r_seq: data.r_validation[batch_ids],
                                             mask: data.mask_validation[batch_ids]})

                x_te_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.x_validation[batch_ids],
                                                              loss_recons: x_recons_te})
                a_te_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.a_validation[batch_ids],
                                                              loss_recons: a_recons_te})
                r_te_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.r_validation[batch_ids],
                                                              loss_recons: r_recons_te})

                validation_nll.append(nll_te)
                validation_kl.append(kl_dist_te)
                validation_x_loss.append(x_te_loss)
                validation_a_loss.append(a_te_loss)
                validation_r_loss.append(r_te_loss)

                elapsed_time = time.time() - start_time

                print('epoch: {:d}, itr: {:d}, nll_tr: {:f}, x_tr_loss: {:f}, a_tr_loss: {:f}, r_tr_loss: {:f}, '
                      'elapsed_time: {:f};'.format(epoch, itr, nll_tr, x_tr_loss, a_tr_loss, r_tr_loss, elapsed_time))

                counter = counter + 1

                if counter % self.opts['plot_every'] == 0:
                    x_0_sample_value = np.reshape(data.x_validation[batch_ids][:,0,:],
                                                [self.opts['batch_size'], 1, self.opts['x_dim']])
                    x_seq_sampling = self.sess.run(x_seq_sample, feed_dict={x_0_sample: x_0_sample_value})

                    filename = 'result_plot_epoch_{:d}_itr_{:d}.png'.format(epoch, itr)

                    save_mnist_plots(self.opts, data.x_train[batch_ids],
                                     data.x_validation[batch_ids],
                                     x_recons_tr, x_recons_te, train_nll, train_kl, validation_nll, validation_kl,
                                     train_x_loss, validation_x_loss, train_a_loss, validation_a_loss, train_r_loss,
                                     validation_r_loss, x_seq_sampling, filename)

        self.saver.save(self.sess,
                        os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_nodecon'),
                        global_step=counter)

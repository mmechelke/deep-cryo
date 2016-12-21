import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


class VariationalAutoencoder(object):

    DEFAULTS = {
        "batch_size" : 128,
        "learning_rate" : 1e-4,
        "non_linearity": tf.nn.elu,
        "image_size" : 32*32,
        "n_hidden_enc1" : 500,
        "n_hidden_enc2" : 100,
        "n_hidden_dec1" : 100,
        "n_hidden_dec2" : 500,
        "n_latent" : 2,
    }

    def __init__(self, n_latent=2):

        self.__dict__.update(VAE.DEFAULTS)
        self.session = tf.Session()

        handles = self.build_graph()


    def _build_graph(self):

        x_in = tf.placeholder(tf.float32,
                              shape=[None, self._params["image_size"]],
                              name="x")

        self.x_in
        self._build_encoder()
        self._sample_z()
        self._build_decoder()

    def _build_encoder(self):
        x_in = self.x_in

        w1_enc = tf.Variabel(xavier_init(self.n_image_size, self.n_hidden_enc1))
        b1_enc = tf.Variable(tf.zeros([self.n_hidden_enc1], dtype=tf.float32))

        w2_enc = tf.Variabel(xavier_init(self.n_hidden_enc1,self.n_hidden_enc2))
        b2_enc = tf.Variable(tf.zeros(self.n_hidden_enc2, dtype=tf.float32))

        w_enc_out = tf.Variabel(xavier_init(self.n_hidden_enc2, self.n_latent))
        b_enc_out = tf.Variable(tf.zeros(self.n_hidden_enc2))

        w_enc_out_sigma = tf.Variabel(xavier_init(self.n_hidden_enc2, self.n_latent))
        b_enc_out_sigma = tf.Variable(tf.zeros(self.n_hidden_enc2))

        enc1 = self.non_linearity(tf.add(tf.matmul(x_in, w1_enc), b1_enc))
        enc2 = self.non_linearity(tf.add(tf.matmul(enc1, w2_enc), b2_enc))

        z_mean = tf.add(tf.matmul(enc2, w_enc_out), b_enc_out)
        z_log_sigma = tf.add(tf.matmul(enc2, w_enc_out_sigma), b_enc_out_sigma)

        self.z_mean, self.z_log_sigma = z_mean, z_log_sigma

    def _sample_z(self):
        mu, log_sigma_sq = self.z_mean, self.z_log_sigma
        epsilon = tf.random_normal(tf.shape(log_sigma_sq), name="epsilon")
        self.z = mu + epsilon * tf.exp(log_sigma_sq)


    def _build_decoder(self,):
        z = self.z
        #Hidden layers
        w1_dec = tf.Variabel(xavier_init(self.n_latent, self.n_hidden_dec1))
        b1_dec = tf.Variable(tf.zeros(self.n_hidden_dec1,))

        w2_dec = tf.Variabel(xavier_init(self.n_hidden_dec1,self.n_hidden_dec2))
        b2_dec = tf.Variable(tf.zeros(self.n_hidden_dec2,))

        #Output layer
        w_dec_out = tf.Variabel(xavier_init(self.n_hidden_enc2, self.image_size))
        b_dec_out = tf.Variable(tf.zeros(self.image_size))

        x_rec = tf.add(tf.matmul(dec2, w_dec_out), b_dec_out)
        self.x_rec = x_rec
        # ops to sample z and explore the latent space
        with tf.name_scope("input_z"):
            z_in = tf.placehoder_with_default(tf.random_normal([1, self.n_latent]),
                                              shape = [None, self.n_latent],
                                              name= "input_z")
        dec1 = self.non_linearity(tf.add(tf.matmul(z_in, w1_de), b1_dec))
        dec2 = self.non_linearity(tf.add(tf.matmul(dec1, w2_dec), b2_dec))
        self._x_rec_z = tf.add(tf.matmul(dec2, w_dec_out), b_dec_out)

    def _create_loss_and_optimizer(self):
        reconstr_loss = \
            -tf.reduce_sum(self.x_in * tf.log(1e-10 + self.x_rec)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_rec),
                           1)

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma), 1)
        self.elbo = tf.reduce_mean(reconstr_loss + latent_loss)

        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = adam.minimize(self.elbo)


    def fit(self, x):

        opt, elbo = self.sess.run((self.optimizer, self.elbo),
                                  feed_dict={self.x_in: x})
        return elbo

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt



def xavier_init(fan_in, fan_out, constant=1.): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


class VariationalAutoencoder(object):

    DEFAULTS = {
        "batch_size" : 128,
        "learning_rate" : 1e-4,
        "non_linearity": tf.nn.elu,
        "image_size" : 784,
        "n_hidden_enc1" : 500,
        "n_hidden_enc2" : 100,
        "n_hidden_dec1" : 100,
        "n_hidden_dec2" : 500,
        "n_latent" : 2,
    }

    def __init__(self, n_latent=2):

        self.__dict__.update(VariationalAutoencoder.DEFAULTS)
        self.session = tf.Session()
        self.handles = self._build_graph()
        self._create_loss_and_optimizer()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def _build_graph(self):

        x_in = tf.placeholder(tf.float32,
                              shape=[None, self.image_size],
                              name="x")

        self.x_in = x_in
        self._build_encoder()
        self._sample_z()
        self._build_decoder()

    def _build_encoder(self):
        x_in = self.x_in

        w1_enc = tf.Variable(xavier_init(self.image_size, self.n_hidden_enc1))
        b1_enc = tf.Variable(tf.zeros([self.n_hidden_enc1], dtype=tf.float32))

        w2_enc = tf.Variable(xavier_init(self.n_hidden_enc1,self.n_hidden_enc2))
        b2_enc = tf.Variable(tf.zeros(self.n_hidden_enc2, dtype=tf.float32))

        w_enc_out = tf.Variable(xavier_init(self.n_hidden_enc2, self.n_latent))
        b_enc_out = tf.Variable(tf.zeros(self.n_latent))

        w_enc_out_sigma = tf.Variable(xavier_init(self.n_hidden_enc2, self.n_latent))
        b_enc_out_sigma = tf.Variable(tf.zeros(self.n_latent))

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
        w1_dec = tf.Variable(xavier_init(self.n_latent, self.n_hidden_dec1))
        b1_dec = tf.Variable(tf.zeros(self.n_hidden_dec1,))

        w2_dec = tf.Variable(xavier_init(self.n_hidden_dec1,self.n_hidden_dec2))
        b2_dec = tf.Variable(tf.zeros(self.n_hidden_dec2,))

        #Output layer
        w_dec_out = tf.Variable(xavier_init(self.n_hidden_dec2, self.image_size))
        b_dec_out = tf.Variable(tf.zeros(self.image_size))

        dec1 = self.non_linearity(tf.add(tf.matmul(z, w1_dec), b1_dec))
        dec2 = self.non_linearity(tf.add(tf.matmul(dec1, w2_dec), b2_dec))
        
        x_rec = tf.nn.sigmoid(tf.add(tf.matmul(dec2, w_dec_out), b_dec_out))

        self.x_rec = x_rec
        # ops to sample z and explore the latent space
        with tf.name_scope("input_z"):
            z_in = tf.placeholder_with_default(tf.random_normal([1, self.n_latent]),
                                              shape = [None, self.n_latent],
                                              name= "input_z")
        dec1_z = self.non_linearity(tf.add(tf.matmul(z_in, w1_dec), b1_dec))
        dec2_z = self.non_linearity(tf.add(tf.matmul(dec1_z, w2_dec), b2_dec))
        self._x_rec_z = tf.add(tf.matmul(dec2_z, w_dec_out), b_dec_out)


    def _create_loss_and_optimizer(self):
        reconstr_loss = \
            -tf.reduce_sum(self.x_in * tf.log(1e-10 + self.x_rec)
                           + (1-self.x_in) * tf.log(1e-10 + 1 - self.x_rec),
                           1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma), 1)
        self.elbo = tf.reduce_mean(reconstr_loss + latent_loss)
        self.reconstr_loss = reconstr_loss
        self.latent_loss = latent_loss

        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        self.optimizer = adam.minimize(self.elbo)


    def fit(self, x):

        opt, elbo = self.session.run((self.optimizer, self.elbo),
                                    feed_dict={self.x_in: x})
        return elbo


if __name__ == "__main__":
    batch_size = 200
    training_epochs  = 10
    vae = VariationalAutoencoder()

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            cost = vae.fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
        print("Epoch:", '%04d'.format(epoch+1), 
              "cost=", "{:.9f}".format(avg_cost))
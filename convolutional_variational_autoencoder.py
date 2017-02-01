import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib import layers

from variational_autoencoder import VariationalAutoencoder

class ConvolutionalVariationalAutoencoder(VariationalAutoencoder):

    DEFAULTS = {
        "batch_size" : 128,
        "learning_rate" : 1e-4,
        "non_linearity": tf.nn.elu,
        "image_size" : 784,
        "n_latent" : 2,
    }

    def __init__(self, n_latent=2):

        self.__dict__.update(ConvolutionalVariationalAutoencoder.DEFAULTS)
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
        net = tf.reshape(x_in, [-1, 28, 28, 1])

        net = layers.conv2d(net, 32, 5, stride=2,
                            activation_fn=tf.nn.elu,
                            normalizer_fn=layers.batch_norm)

        net = layers.conv2d(net, 64, 5, stride=2,
                            activation_fn=tf.nn.elu,
                            normalizer_fn=layers.batch_norm)

        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID',
                            activation_fn=tf.nn.elu,
                            normalizer_fn=layers.batch_norm)

        net = layers.flatten(net)
        encoded = layers.fully_connected(net, 2 * self.n_latent,
                                         activation_fn=None)
        z_mean = encoded[:,:self.n_latent]
        z_log_sigma = encoded[:,self.n_latent:]

        self.z_mean, self.z_log_sigma = z_mean, z_log_sigma


    def _build_decoder(self,):
        z = self.z
        #Hidden layers
        net = tf.expand_dims(z, 1)
        net = tf.expand_dims(net, 1)
        net = layers.conv2d_transpose(net, 128, 3, activation_fn=tf.nn.elu,
                                      normalizer_fn=layers.batch_norm,
                                       padding='VALID')

        net = layers.conv2d_transpose(net, 64, 5, activation_fn=tf.nn.elu,
                                      normalizer_fn=layers.batch_norm,
                                      padding='VALID')

        net = layers.conv2d_transpose(net, 32, 5, activation_fn=tf.nn.elu,
                                      normalizer_fn=layers.batch_norm,
                                      stride=2)

        net = layers.conv2d_transpose(
            net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
        net = layers.flatten(net)

        self.x_rec = net



if __name__ == "__main__":
    batch_size = 200
    training_epochs  = 100
    vae = ConvolutionalVariationalAutoencoder()

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
        print("Epoch: %{}".format(epoch+1) + 
              "cost=", "{:.9f}".format(avg_cost))

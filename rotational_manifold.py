from variational_autoencoder import VariationalAutoencoder

from utils import batch_generator
import numpy as np
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import input_data

number = 1
n_data = 20000
batch_size = 200
training_epochs = 10

vae = VariationalAutoencoder()


# generate data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

data = mnist[0].images
labels = mnist[0].labels

number_ones = data[labels==1]

a_number = number_ones[0].reshape((28,28))
# Random numbers in the range [0,360]
orientations = np.random.random(size=n_data) * 360

data = np.array([ndimage.rotate(a_number, angle, reshape=False).ravel() 
                 for angle in orientations])


for epoch in range(training_epochs):
    avg_cost = 0.
    # Loop over all batches
    for batch_xs in batch_generator(data, batch_size=batch_size):
        # Fit training using batch data
        cost = vae.fit(batch_xs)
        # Compute average loss
        avg_cost += cost / (200/n_data) * batch_size
    print("Epoch: %{}".format(epoch+1) + 
            "cost=", "{:.9f}".format(avg_cost))





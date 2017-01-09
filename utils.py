import numpy as np


def batch_generator(data, labels=None, batch_size=256):
    number_of_batches = np.ceil(data.shape[0]/batch_size)
    sample_index = np.arange(data.shape[0])
    np.random.shuffle(sample_index)
    have_labels = labels is not None
    batch_counter = 0
    while True:
        batch_index = sample_index[batch_size*batch_counter:batch_size*(batch_counter+1)]
        batch = data[batch_index,:]
        if have_labels:
            labels_batch = labels[batch_index]
            batch_counter += 1
            yield batch, labels_batch
        else:
            batch_counter += 1
            yield batch
        if batch_counter >= number_of_batches:
            np.random.shuffle(sample_index)


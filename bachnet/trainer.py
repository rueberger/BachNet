""" This module contains methods for training the model
"""

import time

from bachnet.preprocessing import musicnet_generator
from wavenet.models import Model


SAMPLE_RATE = 44100


def train(n_seconds=30, batch_size=10, epoch_size=1000, n_epochs=1000):
    """ Train a model on n_seconds patches
    """
    n_time_samples = SAMPLE_RATE * n_seconds

    print("Building model")

    model = Model(n_time_samples)

    batch_generator = musicnet_generator(n_time_samples,  batch_size=batch_size,
                                         epoch_size=epoch_size, n_epochs=n_epochs)

    print("Beginning training!")
    # epochs only in name, for now
    for train_idx, (raw_wav, discr_wav, metadata) in enumerate(batch_generator):
        st_time = time.time()
        loss = model._train(raw_wav, discr_wav)
        end_time = time.time()

        if train_idx % epoch_size == 0:
            print("Achieved loss of {}. Training took {}".format(loss, end_time - st_time))

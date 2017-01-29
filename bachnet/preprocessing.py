""" This module contains data preprocessing methods for BachNet"""

import h5py
import csv

import numpy as np


def musicnet_generator(data_dir='/Volumes/ext/data/musicnet',
                       batch_size=10, epoch_size=1000, n_epochs=100):
    """ Generator over MusicNet database and metadata
    """
    metadata_dict = parse_musicnet_metadata('{}/metadata.csv'.format(data_dir))
    data_file = '{}/data.h5'.format(data_dir)

    ids = np.array(metadata_dict)

    with h5py.File(data_file) as data:
        for epoch_idx in range(n_epochs):
            for batch_idx in range(epoch_size):
                row_dict = metadata_dict[batch_idx]
                batch_id = np.random.choice(ids)
                data_row = data['id_{}'.format(batch_id)]['data'][:]
                row_dict['raw_wav'] = data_row
                row_dict['discr_wav'] = discretize_waveform(data_row)
                yield row_dict


def parse_musicnet_metadata(metadata_file):
    """ Parse the musicnet metadata file and parse it into a dictionary keyed by id
    """
    with open(metadata_file, 'r') as md_file:
        md_reader = csv.reader(md_file)
        header = next(md_reader)
        metadata_dict = {}
        for md_row in md_reader:
            row_id = int(md_row[0])
            metadata_dict[row_id] = {label: val for label, val in zip(header[1:], md_row[1:])}
    return metadata_dict

def discretize_waveform(waveform):
    """ Discretize the waveform
    First normalizes the waveform to [-1, 1] then uniformly discretizes into 256 bins
    """
    bins = np.linspace(-1, 1, 256)
    assert waveform.ndim == 1
    norm_waveform = (((np.float32(waveform) - np.min(waveform)) / np.max(waveform)) - 0.5) * 2
    d_waveform = (np.digitize(norm_waveform, bins) - 1).astype(int)
    return d_waveform

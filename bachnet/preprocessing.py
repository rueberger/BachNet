""" This module contains data preprocessing methods for BachNet"""


import os
import h5py
import csv

import numpy as np

WORKSTATION_DATA_PATH = os.path.expanduser('~/data/musicnet')

def musicnet_generator(n_time_samples, data_dir=WORKSTATION_DATA_PATH,
                       batch_size=10, epoch_size=1000, n_epochs=100):
    """ Generator over MusicNet database and metadata
    """
    metadata_dict = parse_musicnet_metadata('{}/metadata.csv'.format(data_dir))
    data_file = '{}/data.h5'.format(data_dir)

    ids = np.array(metadata_dict)

    with h5py.File(data_file) as data:
        for epoch_idx in range(n_epochs):
            for batch_idx in range(epoch_size):
                batch_ids = np.random.choice(ids, size=batch_size)
                sample_dicts = [metadata_dict[id] for id in batch_ids]
                raw_wavs = [data['id_{}'.format(id)]['data'][:] for id in batch_ids]
                discr_wavs = [discretize_waveform(wav) for wav in raw_wavs]

                raw_wav_slices = []
                discr_wav_slices = []
                for raw_wav, discr_wav in zip(raw_wavs, discr_wavs):
                    slice_idx = np.random.randint(0,  len(raw_wav) - n_time_samples - 1)
                    raw_wav_slices.append(raw_wav[slice_idx: slice_idx + n_time_samples])
                    discr_wav_slices.append(discr_wav[slice_idx: slice_idx + n_time_samples])

                # [batch_size, n_time_samples]
                raw_wav_batch = np.stack(raw_wav_slices)
                discr_wav_batch = np.stack(discr_wav_slices)
                yield raw_wav_batch, discr_wav_batch, sample_dicts


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

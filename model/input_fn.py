"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

import model.lfw_dataset as lfw_dataset


def train_input_fn(data_dir, params):
    """Train input function for the LFW dataset.
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = lfw_dataset.get_dataset(data_dir,mode='train')
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the LFW dataset.
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = lfw_dataset.get_dataset(data_dir,mode='test')
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

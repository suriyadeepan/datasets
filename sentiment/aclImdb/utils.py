from random import sample
import numpy as np

import configparser
import os


def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx], y[sample_idx]


def get_config(filename):
    parser = configparser.ConfigParser()
    parser.read(filename)
    conf_ints = [ (key, int(value)) for key,value in parser.items('int') ]
    conf_floats = [ (key, float(value)) for key,value in parser.items('float') ]
    conf_strings = [ (key, str(value)) for key,value in parser.items('str') ]
    return dict(conf_ints + conf_floats + conf_strings)


def assert_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def isEmpty(folder):
    return not (len(os.listdir(folder)) > 0)


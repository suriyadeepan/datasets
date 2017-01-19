import data
import model
import utils

import tensorflow as tf

if __name__ == '__main__':

    metadata, idx_x, idx_y = data.load_data()

    # params
    seqlen = metadata['max_words']
    state_size = 128
    vocab_size = len(metadata['idx2w'])
    batch_size = 128
    num_classes = 2

    train_set = utils.rand_batch_gen(idx_x, idx_y, batch_size)

    smodel = model.SentiRNN(seqlen=seqlen,
                           vocab_size=vocab_size,
                           num_classes=num_classes,
                           num_layers=1,
                           state_size=128,
                           epochs=10000000,
                           learning_rate=0.1,
                           batch_size=batch_size,
                           ckpt_path='ckpt/')

    smodel.train(train_set)

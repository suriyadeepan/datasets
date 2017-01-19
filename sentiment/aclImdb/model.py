import tensorflow as tf
import numpy as np

import sys


class SentiRNN(object):

    def __init__(self, seqlen, vocab_size, num_classes,
            num_layers, state_size, epochs, 
            learning_rate, batch_size, ckpt_path,
            model_name='senti_model'):

        # attach to object
        self.epochs = epochs
        self.state_size = state_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.seqlen = seqlen

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            x_ = tf.placeholder(tf.int32, [None, seqlen], name = 'x')
            y_ = tf.placeholder(tf.int32, [None,], name = 'y')

            # embeddings
            embs = tf.get_variable('emb', [vocab_size, state_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, x_)

            # rnn cell
            cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
            # uncomment line below for increasing depth
            #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
            init_state = cell.zero_state(batch_size, tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)

            # parameters for softmax layer
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], 
                                initializer=tf.constant_initializer(0.0))

            # output for each time step
            y_reshaped = tf.transpose(rnn_outputs, perm=[1,0,2])[-1]
            logits = tf.matmul(y_reshaped, W) + b
            predictions = tf.nn.softmax(logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(0.1).minimize(loss)


            # attach symbols to object, to expose to user of class
            self.x = x_
            self.y = y_
            self.init_state = init_state
            self.train_op = train_op
            self.loss = loss
            self.predictions = predictions
            self.final_state = final_state

        # run build graph
        sys.stdout.write('<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')


    def train(self, train_set, epochs=None):

        epochs = self.epochs if not epochs else epochs

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            #restore session
            if ckpt and ckpt.model_checkpoint_path:
                sys.stdout.write('\nrestoring saved model : {}\n\n'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sys.stdout.write('\ninit global variables\n\n')
                sess.run(tf.global_variables_initializer())

            train_loss = 0
            for i in range(epochs):
                try:
                    # get batches
                    batchX, batchY = train_set.__next__()
                    # run train op
                    feed_dict = { self.x : batchX, self.y : batchY }
                    _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    sys.stdout.write('\r[{}/1000]'.format(1 + (i%1000)))

                    # append to losses
                    train_loss += train_loss_
                    if i and i % 1000 == 0:
                        print('\n>> Average train loss : {}\n'.format(train_loss/1000))
                        # save model to disk
                        saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)

                        # stop condidtion
                        #if (train_loss/1000) < 0.5:
                        #    print('\n>> Loss {}; Stopping training here at iteration #{}!!'.format(train_loss/1000, i))
                        #    break

                        train_loss = 0


                except KeyboardInterrupt:
                    print('\n>> Interrupted by user at iteration #' + str(i))
                    break

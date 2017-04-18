from tensorflow.examples.tutorials.mnist import input_data

# read from disk
mnist_data = input_data.read_data_sets("./")

# get a batch
images, labels = mnist_data.train.next_batch(batch_size=8)

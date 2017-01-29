import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(s, q, a, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(a)
    lens = [ int(data_len*item) for item in ratio ]

    trainS, trainQ, trainA = s[:lens[0]], q[:lens[0]], a[:lens[0]]
    testS, testQ, testA = s[lens[0]:lens[0]+lens[1]],
        q[lens[0]:lens[0]+lens[1]], 
        a[lens[0]:lens[0]+lens[1]]
    validS, validQ, validA = s[-lens[-1]:], q[-lens[-1]:], a[-lens[-1]:]

    return (trainS, trainQ, trainA), 
            (testS, testQ, testA), 
            (validS, validQ, validA)


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(s, q, a, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(a))), batch_size)
        yield s[sample_idx], q[sample_idx], a[sample_idx]


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])

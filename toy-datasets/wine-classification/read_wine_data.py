import numpy as np


def read_wine_data(filename='./wine.txt'):

    def process(split_line):
        x, y = [], []
        for i, item in enumerate(split_line):
            if i > 12:
                y.append(int(item))
            else:
                x.append(float(item))
        return x,y

    def read_file():
        with open(filename, 'r') as f:
            return [ process(line.split('\t')) for line in f.read().split('\n')[:-1] ]

    content = read_file()

    X, Y = [], []
    for x, y in content:
        X.append(x)
        Y.append(y)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

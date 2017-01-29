datadir = '/home/suriya/_/tf/datasets/qa/bAbI/tasks/en/'



import os
import nltk
import itertools



def get_files():
    # get list of files
    filenames = [ fname for fname in os.listdir(datadir) ] 
    # sort them
    filenames = sorted(filenames,
            key = lambda x : int( x[2 : x.find('_')] ))
    # add path to filename
    return [ datadir + fname for fname in filenames ]


def read_task(task_id):
    # get file names
    filenames = get_files()
    # get task specific files
    tfilenames = filenames[(task_id-1)*2 : (task_id-1)*2 + 2]
    # read from files
    content = ''
    for filename in tfilenames:
        with open(filename) as f:
            content += f.read()

    return content.split('\n')[:-1]


def reshape_content(lines):
    rows = []
    stories = []
    for line in lines:
        if '?' in line:
            answer = line.split(' ')[-1].split('\t')[1]
            question = ' '.join(line.split(' ')[1:])
            question = question[ : question.find('?') + 1 ]
            rows.append([stories, question, answer]) 
            # empty(v) stories
            stories = []
        else:
            stories.append( ' '.join(line.split(' ')[1:]) ) 
    return rows


def index_(tokenized_sentences):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common()
    # index2word
    index2word = ['_'] + [ x[0] for x in vocab ] # - : zero paddding
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def tokenize(lines):
    tokenized = []
    for line in lines:
        words = []
        for word in line.split(' '):
            words.append( ''.join(
                [ ch for ch in word if ch.isalpha() ] ))
        tokenized.append(words)
    return tokenized


def zero_pad(data, w2i):





def process_data():
    lines = read_task(1)
    data = reshape_content(lines)
    # index data
    tokenized = tokenize(lines)
    i2w, w2i, freq_dist = index_(tokenized)
    print(i2w)



if __name__ == '__main__':
    process_data()

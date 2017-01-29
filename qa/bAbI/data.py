datadir = '/home/suriya/_/tf/datasets/qa/bAbI/tasks/en-10k/'
WHITELIST = 'abcdefghijklmnopqrstuvwxyz, '



import os
import nltk
import itertools
import numpy as np
import pickle
from random import shuffle



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
            content += f.read().lower()

    return content.split('\n')[:-1]

def read_all_tasks():
    # get file names
    filenames = get_files()
    # read from files
    content = ''
    for filename in filenames:
        with open(filename) as f:
            content += f.read().lower()

    return content.split('\n')[:-1]


def reshape_content(lines):
    stories = []
    story_lines = []
    questions = []
    answers = []
    for line in lines:
        if line and '?' in line:
            # parse answer
            try:
                answer = line[line.find('?')+1:].split('\t')[1]
            except:
                print(line)
            # parse question
            question = ' '.join(line.split(' ')[1:])
            question = question[ : question.find('?') + 1 ]
            # add items to lists
            stories.append(story_lines)
            questions.append(filter_word(question))
            answers.append(filter_word(answer))
            # empty(v) story_lines
            story_lines = []
        else:
            if line:
                # add lines of each story(facts) to temp list
                story_lines.append( filter_word(' '.join(line.split(' ')[1:]) ) )
    return stories, questions, answers


def index_(tokenized_sentences):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common()
    # index2word
    index2word = ['_'] + ['?'] + ['.'] + [ x[0] for x in vocab ] # - : zero paddding
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def tokenize(lines):
    tokenized = []
    for line in lines:
        words = []
        for word in line.split(' '):
            words.append( ''.join(
                [ ch for ch in word if ch in WHITELIST ] ))
        tokenized.append(words)
    return tokenized


def filter_word(word):
    return ''.join([ ch for ch in word if ch in WHITELIST ])

def zero_pad(stories, questions, answers, w2i):
    num_lines = len(answers)
    # answers
    idx_a = np.array([ w2i[ans] for ans in answers ], dtype=np.int32)
    # questions
    max_q_sent_len = max([ len(q.split(' ')) for q in questions ])
    idx_q = np.zeros([num_lines, max_q_sent_len + 1], dtype=np.int32)
    # iterate through questions
    for i,q in enumerate(questions):
        for j,w in enumerate(q.split(' ')):
            idx_q[i][j] = w2i[w.replace('?', '')]
        # add ? to question
        idx_q[i][j+1] = w2i['?']
    # stories
    max_s_sent_len = max([ len(line.split(' ')) 
        for story in stories for line in story ])
    max_s_lines = max([ len(story) for story in stories ])
    idx_s = np.zeros([num_lines, max_s_lines, max_s_sent_len + 1], 
            dtype=np.int32)

    for i,story in enumerate(stories):
        for j,line in enumerate(story):
            for k,w in enumerate(line.split(' ')):
                idx_s[i][j][k] = w2i[w.replace('.', '')]
            idx_s[i][j][k+1] = w2i['.']

    return idx_a, idx_q, idx_s


def process_data():
    lines = read_all_tasks()
    stories, questions, answers = reshape_content(lines)
    # shuffle data
    data = list(zip(stories, questions, answers))
    shuffle(data)
    stories, questions, answers = zip(*data)

    # index data
    tokenized = [ w for story in stories  
            for line in story for w in line.split(' ') ]
    tokenized += [ w for q in questions for w in q.split(' ') ]
    tokenized += [ w for w in answers] 

    # get loopups
    idx2w, w2idx, freq_dist = index_(tokenized)

    #  zero padding
    idx_a, idx_q, idx_s = zero_pad(stories, questions, answers, w2idx)

    print('shapes')
    print(idx_a.shape, idx_q.shape, idx_s.shape)

    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_s.npy', idx_s)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)



if __name__ == '__main__':
    process_data()


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_s = np.load(PATH + 'idx_s.npy')
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_s, idx_q, idx_a

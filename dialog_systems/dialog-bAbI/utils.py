DATA_SOURCE = 'data/dialog-bAbI-tasks/dialog-babi-candidates.txt'
DATA_SOURCE_TASK6 = 'data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-candidates.txt'
DATA_DIR = 'dialog-bAbI-tasks/dialog-babi-candidates.txt'
STOP_WORDS=set(["a","an","the"])


import re
import os

from itertools import chain
from six.moves import range, reduce

import numpy as np
import tensorflow as tf


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in STOP_WORDS]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result


def load_candidates(task_id, candidates_f=DATA_SOURCE):
    # containers
    candidates, candid2idx, idx2candid = [], {}, {}
    # update data source file based on task id
    candidates_f = DATA_SOURCE_TASK6 if task_id==6 else candidates_f
    # read from file
    with open(candidates_f) as f:
        # iterate through lines
        for i, line in enumerate(f):
            # tokenize each line into... well.. tokens!
            candid2idx[line.strip().split(' ',1)[1]] = i
            candidates.append(tokenize(line.strip()))
            idx2candid[i] = line.strip().split(' ',1)[1]
    return candidates, candid2idx, idx2candid


def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    u=None
    r=None
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if '\t' in line:
                u, r = line.split('\t')
                a = candid_dic[r]
                u = tokenize(u)
                r = tokenize(r)
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                data.append((context[:],u[:],a))
                u.append('$u')
                u.append('#'+str(nid))
                r.append('$r')
                r.append('#'+str(nid))
                context.append(u)
                context.append(r)
            else:
                r=tokenize(line)
                r.append('$r')
                r.append('#'+str(nid))
                context.append(r)
        else:
            # clear context
            context=[]
    return data


def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)


def load_dialog_task(data_dir, task_id, candid_dic, isOOV=False):
    '''Load the nth task. 
    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def build_vocab(data, candidates, memory_size=50):
    vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in data))
    vocab |= reduce(lambda x,y: x|y, (set(candidate) for candidate in candidates) )
    vocab=sorted(vocab)
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    candidate_sentence_size=max(map(len,candidates))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(w2idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position

    return {
            'w2idx' : w2idx,
            'idx2w' : vocab,
            'sentence_size' : sentence_size,
            'candidate_sentence_size' : candidate_sentence_size,
            'memory_size' : memory_size,
            'vocab_size' : vocab_size,
            'n_cand' : len(candidates)
            } # metadata


def vectorize_candidates(candidates, word_idx, sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)


def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, answer) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq

        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
    return S, Q, A


def get_batches(train_data, val_data, test_data, metadata, batch_size):
    '''
    input  : train data, valid data
        metadata : {batch_size, w2idx, sentence_size, num_cand, memory_size}
    output : batch indices ([start, end]); train, val split into stories, ques, answers

    '''
    w2idx = metadata['w2idx']
    sentence_size = metadata['sentence_size']
    memory_size = metadata['memory_size']
    n_cand = metadata['n_cand']

    trainS, trainQ, trainA = vectorize_data(train_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    valS, valQ, valA = vectorize_data(val_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    testS, testQ, testA = vectorize_data(test_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    n_train = len(trainS)
    n_val = len(valS)
    n_test = len(testS)
    print("Training Size",n_train)
    print("Validation Size", n_val)
    print("Test Size", n_test)
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))

    # package train set 
    train = { 's' : trainS, 'q' : trainQ, 'a' : trainA } # you have a better idea?
    # package validation set 
    val =   { 's' : valS, 'q' : valQ, 'a' : valA } 
    # package test set 
    test =   { 's' : testS, 'q' : testQ, 'a' : testA }
 
    return train, val, test, [(start, end) for start, end in batches]



if __name__ == '__main__':
    candidates, candid2idx, idx2candid = load_candidates(task_id=1)

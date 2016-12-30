import nltk


def combinations(word, blacklist):
    combs = []
    for i in range(len(word)):
        if word[i] not in blacklist:
            for j in range(i+2,len(word)+1):
                combs.append(word[i:j])
    return combs

def greedy_combinations(word):
    combs = []
    wlen = len(word)
    return [ word[:wlen-i] for i in range(len(word)) ]

def freq_combinations(lines, blacklist):

    symbols = []
    for line in lines:
        for word in line.split(' '):
            symbols.extend(combinations(word, blacklist))

    return nltk.FreqDist(symbols)

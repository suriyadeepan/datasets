'''
This program plots the lengths of source input and target pairs.

The intention is for one to use this to help determine bucket sizes.

Maybe in the future I will implement a clustering algorithm to autonomously find
bucket sizes
'''



import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
from prepare_data import get_id2line, get_conversations, gather_dataset

num_bins = 50
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def main():

    id2line = get_id2line()
    print '>> gathered id2line dictionary.\n'
    convs = get_conversations()
    print '>> gathered conversations.\n'
    questions, answers = gather_dataset(convs,id2line)

    target_lengths = [ len(basic_tokenizer(line)) for line in answers]
    source_lengths = [ len(basic_tokenizer(line)) for line in questions]

	#if FLAGS.plot_histograms:
    plotHistoLengths("target lengths", target_lengths)
    plotHistoLengths("source_lengths", source_lengths)
    plotScatterLengths("target vs source length", "source length", "target length", source_lengths, target_lengths)


def plotScatterLengths(title, x_title, y_title, x_lengths, y_lengths):
	plt.scatter(x_lengths, y_lengths)
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	#plt.ylim(0, max(y_lengths))
	#plt.xlim(0,max(x_lengths))
	plt.ylim(0, 200)
	plt.xlim(0, 200)
	plt.show()

def plotHistoLengths(title, lengths):
	mu = np.std(lengths)
	sigma = np.mean(lengths)
	x = np.array(lengths)
	n, bins, patches = plt.hist(x,  num_bins, facecolor='green', alpha=0.5)
	y = mlab.normpdf(bins, mu, sigma)
	plt.plot(bins, y, 'r--')
	plt.title(title)
	plt.xlabel("Length")
	plt.ylabel("Number of Sequences")
	plt.xlim(0,80)
	plt.show()


if __name__=="__main__":
	main()

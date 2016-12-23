# CMU Pronouncing Dictionary


The CMU Pronouncing Dictionary is an open source pronouncing dictionary originally created by the Speech Group at Carnegie Mellon University (CMU) for use in speech recognition research. CMUdict provides a mapping orthopraphic/phonetic for English words in their North American pronunciations. 


## Raw Data

Download the raw data from [here](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)


## Processed Data

Processed data *cmudict-processed.tar.gz* contains

- data_ctl.pkl : { idx2alpha, idx2pho, pho2idx, alpha2idx, limit }
	- limit : { maxw, minw, maxph, minph } (upper and lower limits to sequence lengths)
- idx_phonemes.npy : array of indices of phonemes
- idx_words.npy : array of indices of characters in words


## Script

The script **data.py**, reads the raw data (cmudict-0.7b), creates arrays of indices of phonemes and words, which can be decoded with data control dictionaries (idx2pho, idx2alpha). 

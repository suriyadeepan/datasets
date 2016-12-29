# Twitter Chat log

Borrowed from [Marson-Ma](https://github.com/Marsan-Ma/chat_corpus/)

Download from [here](https://raw.githubusercontent.com/Marsan-Ma/chat_corpus/master/twitter_en.txt.gz)

## Processing

```bash
cd data
./pull
# now extract the .txt.gz file
#  i'm lazy
cd ..
python3 data.py
```

## Processed Data

- [seq2seq.twitter.tar.gz](https://www.dropbox.com/s/a091opu8dhdx1lu/seq2seq.twitter.tar.gz?dl=0)
	- idx\_q.npy
	- idx\_a.npy
	- metadata.pkl
		- w2idx
		- idx2w
		- limit : { maxq, minq, maxa, mina }
		- freq_dist

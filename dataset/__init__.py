from typing import Tuple, List
import math
import random

import numpy
from keras import utils
from dataset.chars import Chars

MAXLEN = 128
TWEETS_PATH = './dataset/tweets.txt'


class Sequence(utils.Sequence):

    def __init__(self, data, indices: List[int], batch_size: int, chars: Chars):
        self.data = data
        self.indices = indices or list(range(len(data)))
        self.batch_size = batch_size
        self.chars = chars

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = begin + self.batch_size
        batch_idx = self.indices[begin: end]
        batch = [self.data[i] for i in batch_idx]

        if type(batch[0]) == str:  # tweets
            X = numpy.array([self.chars.convert(line, sparse=False) for line in batch])
            Y = numpy.array([self.chars.convert(line, sparse=True) for line in batch])
            Y = Y.reshape(Y.shape + (1,))  # fuck
            return X, Y


def get_chars() -> Chars:
    lines = [line.strip() for line in open(TWEETS_PATH)]
    return Chars(lines, MAXLEN)


def load_tweets(batch_size, validation_split=0.1) -> Tuple[Sequence, Sequence]:

    chars = get_chars()

    random.seed(42)
    lines = [line.strip() for line in open(TWEETS_PATH)]
    random.shuffle(lines)

    indices = list(range(len(lines)))
    random.shuffle(indices)

    num_valid = int(len(lines) * validation_split)
    indices_train = indices[num_valid:]
    indices_valid = indices[:num_valid]

    seq_train = Sequence(lines, indices_train, batch_size, chars)
    seq_valid = Sequence(lines, indices_valid, batch_size, chars)
    return seq_train, seq_valid

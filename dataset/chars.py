import collections
import numpy
from typing import List


class Chars():

    @classmethod
    def is_good_char(cls, char: str) -> bool:
        return True

    @classmethod
    def is_too_rare(cls, count: int) -> bool:
        return count <= 1

    def __init__(self, lines: List[str], maxlen: int):

        counter = collections.defaultdict(int)
        for line in lines:
            for c in line:
                if Chars.is_good_char(c):
                    counter[c] += 1

        chars = ['__EMPTY__', '__UNK__', '__BOS__', '__EOS__']
        for c in counter:
            if Chars.is_too_rare(counter[c]):
                continue
            chars.append(c)

        del counter
        self.chars = chars
        self.maxlen = maxlen

    def __getitem__(self, idx: int) -> str:
        return self.chars[idx]

    def __len__(self) -> int:
        return len(self.chars)

    def index(self, char: str) -> int:
        return self.chars.index(char) if char in self.chars else None

    def convert(self, line, sparse=False) -> numpy.array:

        line = line.strip()
        unk_id = self.index('__UNK__')

        if sparse:
            X = numpy.zeros((self.maxlen,)).astype('i')
            for i, c in enumerate(line):
                if i + 2 > self.maxlen:
                    break
                idx = self.index(c)
                X[i + 1] = unk_id if idx is None else idx
            X[0] = self.index('__BOS__')
            X[len(line) + 1] = self.index('__EOS__')
            return X

        else:
            X = numpy.zeros((self.maxlen, len(self.chars))).astype('f')
            for i, c in enumerate(line):
                if i + 2 > self.maxlen:
                    break
                idx = self.index(c)
                X[i + 1][unk_id if idx is None else idx] = 1
            X[0][self.index('__BOS__')] = 1
            X[len(line) + 1][self.index('__EOS__')] = 1
            return X

    def deconvert(self, X: numpy.array) -> List[str]:
        if X.ndim == 1:
            return [self[x] for x in X]
        elif X.ndim == 2:
            return self.deconvert(numpy.argmax(X, axis=1))
        else:
            return None

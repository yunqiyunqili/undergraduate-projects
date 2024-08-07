import numpy as np
import pandas as pd


def dox2onehotmatrix(filename):
    with open(filename) as file:
        docs = file.readlines()
        words = []
        for i in range(len(docs)):
            docs[i] = docs[i].strip().split()
            words += docs[i]
        vocab=sorted(set(words),key=words.index)
        M=len(docs)
        V=len(vocab)
        onehot = np.zeros((M,V))
        for i,doc in enumerate(docs):
            for word in doc:
                if word in vocab:
                    pos = vocab.index(word)
                    onehot[i][pos] = 1
        onehot = pd.DataFrame(onehot, columns = vocab)
        return onehot

onehot = dox2onehotmatrix('data.txt')
print(onehot)
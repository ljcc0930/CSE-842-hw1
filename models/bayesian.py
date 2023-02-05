import numpy as np

import utils


class NaiveBayes:
    def __init__(self, n_corpus, n_class):
        self.n_corpus = n_corpus
        self.n_class = n_class
        self.N_c = np.zeros(n_class, dtype=int)
        self.N_w_c = np.zeros([n_corpus, n_class], dtype=int)
        self.n_doc = 0
        # P_w_c = np.zeros(n_corpus, n_class)

    def update(self, docs, labels):
        for doc, label in zip(docs, labels):
            self.N_c[label] += 1
            for line in doc:
                for word in line:
                    self.N_w_c[word][label] += 1

    def predict(self, docs, k=0):
        P_c = self.N_c / self.N_c.sum()
        P_w_c = (self.N_w_c + k) / (self.N_w_c + k).sum(axis=0)
        lP_c = np.log(P_c)
        lP_w_c = np.log(P_w_c)

        prob = [np.sum(lP_w_c[utils.concat_lists(*doc)], axis=0) + lP_c
                for doc in docs]

        return np.argmax(prob, axis=1)

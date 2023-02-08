import numpy as np

import utils


class NaiveBayes:
    def __init__(self, n_vocabulary, n_class, smooth_k=1):
        self.attrs = ["n_vocabulary", "n_class",
                      "N_c", "N_w_c", "lP_c", "lP_w_c"]
        self.n_vocabulary = n_vocabulary
        self.n_class = n_class
        self.set_smooth_k(smooth_k)
        self.clear()
        # P_w_c = np.zeros(n_vocabulary, n_class)

    def clear(self):
        self.N_c = np.zeros(self.n_class, dtype=int)
        self.N_w_c = np.zeros([self.n_vocabulary, self.n_class], dtype=int)

        self.lP_c = np.zeros(self.n_class)
        self.lP_w_c = np.zeros([self.n_vocabulary, self.n_class])

    def set_smooth_k(self, k):
        self.k = k

    def finalize(self):
        P_c = self.N_c / self.N_c.sum()
        P_w_c = (self.N_w_c + self.k) / (self.N_w_c + self.k).sum(axis=0)

        self.lP_c = np.log(P_c)
        self.lP_w_c = np.log(P_w_c)

    def update(self, docs, labels):
        for doc, label in zip(docs, labels):
            self.N_c[label] += 1
            for word in doc:
                self.N_w_c[word][label] += 1
        self.finalize()

    def predict(self, docs):
        prob = [np.sum(self.lP_w_c[doc],
                       axis=0) + self.lP_c for doc in docs]

        return np.argmax(prob, axis=1)

    def save(self, path, method="pickle"):
        state_dict = {attr: getattr(self, attr) for attr in self.attrs}
        utils.dump_obj(state_dict, path, method)

    def load(self, path, method="pickle"):
        state_dict = utils.load_obj(path, method)
        for attr in self.attrs:
            if attr in state_dict:
                setattr(self, attr, state_dict[attr])

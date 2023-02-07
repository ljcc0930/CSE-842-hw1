import sklearn.naive_bayes


class SklearnNaiveBayes:
    def __init__(self, smooth_k=1):
        self.clear()
        self.set_smooth_k(smooth_k)

    def clear(self):
        self.model = sklearn.naive_bayes.MultinomialNB()

    def set_smooth_k(self, k):
        self.model.alpha = k

    def update(self, docs, labels):
        self.model.fit(docs, labels)

    def predict(self, docs):
        return self.model.predict(docs)

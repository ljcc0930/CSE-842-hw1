import sklearn.svm


class SklearnSvm:
    def __init__(self, c=1):
        self.c = c
        self.clear()

    def clear(self):
        self.model = sklearn.svm.SVC(C=self.c)

    def update(self, docs, labels):
        self.model.fit(docs, labels)

    def predict(self, docs):
        return self.model.predict(docs)

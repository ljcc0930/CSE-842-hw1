from .bayesian import NaiveBayes

def get_model(name, n_corpus, n_class):
    if name == 'naive_bayes':
        return NaiveBayes(n_corpus, n_class)
    else:
        raise NotImplementedError("Model '{}' not implemented!".format(name))
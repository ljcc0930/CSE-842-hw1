from .bayesian import NaiveBayes
from .bayesian_sklearn import SklearnNaiveBayes
from .SVM_sklearn import SklearnSvm


def get_model(name, *args, **kwargs):
    if name == 'naive_bayes':
        return NaiveBayes(*args, **kwargs)
    elif name == 'sklearn_naive_bayes':
        return SklearnNaiveBayes(*args, **kwargs)
    elif name == 'sklearn_svm':
        return SklearnSvm(*args, **kwargs)
    else:
        raise NotImplementedError("Model '{}' not implemented!".format(name))

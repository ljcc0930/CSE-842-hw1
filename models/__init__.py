from .bayesian import NaiveBayes


def get_model(name, n_vocabulary, n_class, *args, **kwargs):
    if name == 'naive_bayes':
        return NaiveBayes(n_vocabulary, n_class, *args, **kwargs)
    else:
        raise NotImplementedError("Model '{}' not implemented!".format(name))

from .polarity import Polarity
from .polarity_sklearn import SklearnPolarity
from .sarcasm import Sarcasm


def get_dataset(name, n_folds=3, n_grams=1, data_dir='data'):
    if name == "polarity":
        return Polarity(n_folds, n_grams, data_dir)
    if name == "sarcasm":
        return Sarcasm(n_folds, n_grams, data_dir)
    if name == "sklearn_polarity":
        return SklearnPolarity(n_folds, n_grams, data_dir)
    else:
        raise NotImplementedError("Dataset '{}' not implemented!".format(name))

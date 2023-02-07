from .polarity import Polarity
from .polarity_sklearn import SklearnPolarity


class Sarcasm:
    def __init__(self):
        self.dataset_url = "http://nldslab.soe.ucsc.edu/sarcasm/sarcasm_v2.zip"


def get_dataset(name, n_folds=3, data_dir='data'):
    if name == "polarity":
        return Polarity(n_folds, data_dir)
    if name == "sklearn_polarity":
        return SklearnPolarity(n_folds, data_dir)
    else:
        raise NotImplementedError("Dataset '{}' not implemented!".format(name))

from .polarity import Polarity


class Sarcasm:
    def __init__(self):
        self.dataset_url = "http://nldslab.soe.ucsc.edu/sarcasm/sarcasm_v2.zip"


def get_dataset(name, n_folds=3, data_dir='data'):
    if name == "polarity":
        return Polarity(n_folds, data_dir)
    else:
        raise NotImplementedError("Dataset '{}' not implemented!".format(name))


if __name__ == "__main__":
    n = 3
    ds = get_dataset("polarity", n_folds=n)
    for fold in range(3):
        train_text, train_label, test_text, test_label = ds.get_datasets(fold)
        print(len(train_text))
        print(len(train_label))
        print(len(test_text))
        print(len(test_label))
    ds.encode()
    ds.decode()

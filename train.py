import utils


def test(model, text, label):
    output = model.predict(text)
    precision, recall, f1, accuracy = utils.evaluation(output, label)
    print("\tPrecision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}.".format(
        precision, recall, f1, accuracy))
    return precision, recall, f1, accuracy


def train(model, train_text, train_label):
    model.update(train_text, train_label)
    test(model, train_text, train_label)

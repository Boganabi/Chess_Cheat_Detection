
# https://towardsdatascience.com/how-to-benefit-from-the-semi-supervised-learning-with-label-spreading-algorithm-2f373ae5de96/
# https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
# https://machinelearningmastery.com/semi-supervised-learning-with-label-spreading/

import numpy as np
from numpy import concatenate

from base_model import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

class LabelSpread(BaseModel):

    def __init__(self):
        self.X_train_label = None
        self.X_test_unlabel = None
        self.y_train_label = None
        self.y_test_unlabel = None

    def start(self):
        self.get_data("../datasets/l_Games.csv")
        self.split_labels()

    def split_labels(self):
        # divide training set into labeled and "unlabeled" data
        self.X_train_label, self.X_test_unlabel, self.y_train_label, self.y_test_unlabel = train_test_split(self.X_train, self.y_train, test_size=0.1)

    def train(self):
        X_train_mixed = concatenate((self.X_train_label, self.X_test_unlabel))
        # nolabel = [-1 for _ in range(len(self.y_test_unlabel))]
        nolabel = np.empty(self.y_test_unlabel.shape)
        nolabel.fill(-1)

        # recombine training dataset labels
        y_train_mixed = concatenate((self.y_train_label, nolabel))
        model = LabelSpreading()
        model.fit(X_train_mixed, y_train_mixed)

        # show accuracy
        yhat = model.predict(self.X_test)
        score = accuracy_score(self.y_test, yhat)
        print("Accuracy: %.3f" % (score*100))

    def summarize(self):
        print("Labeled set:", self.X_train_label.shape, self.y_train_label.shape)
        print("Unlabeled set:", self.X_test_unlabel.shape, self.y_test_unlabel.shape)
        print("Test set:", self.X_test.shape, self.y_test.shape)


if __name__ == "__main__":
    ls = LabelSpread()
    ls.start()
    ls.summarize()
    ls.train()
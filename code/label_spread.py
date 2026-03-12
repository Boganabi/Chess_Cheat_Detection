
# https://towardsdatascience.com/how-to-benefit-from-the-semi-supervised-learning-with-label-spreading-algorithm-2f373ae5de96/
# https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
# https://machinelearningmastery.com/semi-supervised-learning-with-label-spreading/

import numpy as np
import pandas as pd
import joblib

from numpy import concatenate
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack

from base_model import BaseModel

class LabelSpread(BaseModel):

    def __init__(self):
        super().__init__()
        self.X_train_label = None
        self.X_test_unlabel = None
        self.y_train_label = None
        self.y_test_unlabel = None
        self.loaded_model = None
        self.loaded_encoder = None

    def start(self):
        self.get_data("../datasets/l_Games.csv")
        self.split_labels()

    def split_labels(self):
        # divide training set into labeled and "unlabeled" data
        self.X_train_label, self.X_test_unlabel, self.y_train_label, self.y_test_unlabel = train_test_split(self.X_train, self.y_train, test_size=0.1)

    def train(self):
        # X_train_mixed = concatenate((self.X_train_label, self.X_test_unlabel))
        X_train_mixed = vstack([self.X_train_label, self.X_test_unlabel])
        # nolabel = [-1 for _ in range(len(self.y_test_unlabel))]
        nolabel = np.empty(self.y_test_unlabel.shape)
        nolabel.fill(-1)

        # recombine training dataset labels
        y_train_mixed = concatenate((self.y_train_label, nolabel))
        # y_train_mixed = hstack([self.y_train_label, nolabel])
        model = LabelSpreading()
        print("training model...")
        model.fit(X_train_mixed, y_train_mixed)

        # show accuracy
        yhat = model.predict(self.X_test)
        score = accuracy_score(self.y_test, yhat)
        print("Accuracy: %.3f" % (score*100))

        # save model
        joblib.dump(model, "../models/label_spread.pkl")
        joblib.dump(self.encoder, "../models/encoder.pkl")

    def summarize(self):
        print("Labeled set:", self.X_train_label.shape, self.y_train_label.shape)
        print("Unlabeled set:", self.X_test_unlabel.shape, self.y_test_unlabel.shape)
        print("Test set:", self.X_test.shape, self.y_test.shape)

    def load_model(self):
        self.loaded_model = joblib.load("../models/label_spread.pkl")
        self.loaded_encoder = joblib.load("../models/encoder.pkl")

    def test_model(self, game: str, w_elo: int, b_elo: int):
        if(self.loaded_model and self.loaded_encoder):
            df = pd.DataFrame({"Game": [game], "Elo White": [w_elo], "Elo Black": [b_elo]})
            print(self.loaded_model.classes_)
            X = self.loaded_encoder.transform(df)
            return self.loaded_model.predict(X)


if __name__ == "__main__":
    ls = LabelSpread()
    # ls.start()
    # ls.summarize()
    # ls.train()
    ls.load_model()
    GAME = "1. d4 f5 2. c4 Nf6 3. Nc3 e6 4. Qc2 Bb4 5. e3 b6 6. Bd3 Bb7 7. f3 O-O 8. Ne2 Bd6 9. Bd2 Na6 10. a3 c5 11. O-O-O cxd4 12. exd4 Rc8 13. Kb1 Nc7 14. Rhe1 Kh8 15. Bf4 Bxf4 16. Nxf4 b5 17. cxb5 Ncd5 18. Nfxd5 Nxd5 19. Qb3 Qh4 20. Na4 Qxd4 21. Bxf5 Qf2 22. Be4 Qxg2 23. Ka1 Qf2 24. Rf1 Qe2 25. Rfe1 Qf2 26. Rf1 Qe2 27. Rfe1 Qf2 28. Rf1"
    result = ls.test_model(GAME, 2519, 2694)
    print(result)
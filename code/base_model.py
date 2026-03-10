
# functions & variables that all models will use, to be inherited by those specific models

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class BaseModel:
    
    def __init__(self):
        self.X_train
        self.X_test
        self.y_train
        self.y_test
        self.df
        self.encoder = OneHotEncoder(sparse_output=False)

    def get_data(self, csv_data: str):
        self.df = pd.read_csv(csv_data)
        X = self.df[["matrix_game", "Elo White", "Elo Black"]]#, "Score"]]
        y = self.df["cheat_code"] # for a regression model we can change to the by-move labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.1, random_state=1)

    def get_encoding(self, X):
        return self.encoder.fit_transform(X)
    
# this might be interesting to look into https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html 
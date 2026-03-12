
# functions & variables that all models will use, to be inherited by those specific models

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class BaseModel():
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def get_data(self, csv_data: str, encode=True):
        self.df = pd.read_csv(csv_data)
        X = self.df[["Game", "Elo White", "Elo Black"]]#, "Score"]]
        y = self.df["cheat_code"] # for a regression model we can change to the by-move labels

        # encode training data
        if(encode):
            X = self.get_encoding(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.1, random_state=1)

    def get_encoding(self, X):
        return self.encoder.fit_transform(X)
    
# this might be interesting to look into https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html 
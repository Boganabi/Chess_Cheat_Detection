
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack

df = pd.read_csv("../datasets/l_Games.csv")

cat_features = ["Game"]
num_features = ["Elo White", "Elo Black"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough' # Keep other columns if any
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LabelSpreading()) 
])

X = df[["Game", "Elo White", "Elo Black"]]#, "Score"]]
y = df["cheat_code"] # for a regression model we can change to the by-move labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=1)
X_train_label, X_test_unlabel, y_train_label, y_test_unlabel = train_test_split(X_train, y_train, test_size=0.1)

# X_train_mixed = np.concatenate((X_train_label, X_test_unlabel))
# # X_train_mixed = vstack([X_train_label, X_test_unlabel])
# nolabel = np.empty(y_test_unlabel.shape)
# nolabel.fill(-1)

# # recombine training dataset labels
# y_train_mixed = np.concatenate((y_train_label, nolabel))
# pipeline.fit(X_train_mixed, y_train_mixed)

print("training model...")
pipeline.fit(X_train, y_train)

# show accuracy
yhat = pipeline.predict(X_test)
score = accuracy_score(y_test, yhat)
print("Accuracy: %.3f" % (score*100))

# save model
joblib.dump(pipeline, "../models/label_spread_pipeline.pkl") 
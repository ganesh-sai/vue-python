import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Getting and Parsing the data 
training_data = pd.read_csv("iris.data")


# Naming the columns
training_data.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]

# Split up the columns of the dataframe above into features and labels.
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
label_cols = ["class"]

# Format the data
X = training_data.loc[:,feature_cols]
Y = training_data.loc[:, label_cols].values.ravel()

# Split data into test and train sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# train the model
log_reg = LogisticRegression(solver="liblinear",multi_class="ovr")
clf = log_reg.fit(x_train, y_train)

# see if the model is reasonable 
print("Score: ",clf.score(x_test, y_test))

# Pickle to save the model for use in out API
pickle.dump(clf, open("./our_model.pkl","wb"))


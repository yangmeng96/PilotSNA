import pandas as pd
import numpy as np
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import statsmodels.api as sm



# LR is a function to do logistic regression on traning data and prediction on test data.
    # train: training data.
    # test: test data.
    # Y: the outcome. In our setting, it should be "recur".
    # X: all the covariates/predictors, they are comorbidities(all or the first K principal components) the and demographics.
    # For demographics, we only have four predicitors: age, stroke_subtype, sex and race.
    # tolerance: the tolerance value when stopping iteration.
    # seed: the random seed.

def LR(train,test,tolerance,iter,seed):
    Y_train = train["xxx"] # the column name of the outcome
    X_train = train.drop(['xxx'], axis=1) # remove the useless columns
    Y_test = test["xxx"]
    X_test = test.drop(['xxx'], axis=1)

    X_train = pd.get_dummies(X_train, columns=['xxx']) # change to dummy variables
    X_test = pd.get_dummies(X_test, columns=['xxx'])

    model = LogisticRegression(solver='saga', tol = tolerance, max_iter = iter, random_state = seed)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_scores = model.predict_proba(X_test)[:, 1] 
    FN = sum((Y_test == 1) & (Y_pred == 0))
    TP = sum((Y_test == 1) & (Y_pred == 1))
    recall = TP/(FN+TP)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(model.coef_)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    return(Y_scores, accuracy, recall)





  


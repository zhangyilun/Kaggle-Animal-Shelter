# import
import pandas as pd
import numpy as np
import operator
import re
import os

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split


# function to write submission output
def create_submission(pred, test_ID):
    sub = pd.DataFrame(pred)
    sub["ID"] = test_ID
    cols = sub.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    sub = sub[cols]
    sub.columns = ["ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer"]
    return sub

def main():

    # data 
    train = pd.read_csv("data/train_clean_31.csv")
    test = pd.read_csv("data/test_clean_31.csv")

    # for test_clean_31 only
    test = test.replace(np.nan,7691)

    # split
    # set missing
    train_X = train.ix[:, train.columns != "OutcomeType"]
    train_Y = train["OutcomeType"]

    test_ID = test["ID"]
    test_X = test.ix[:, test.columns != "ID"]

    # if using the smote data set
    test_X.columns = [x.replace(" ",".") for x in test_X.columns]

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)

    train_X = train_X.astype(float)
    test_X = test_X.astype(float)

    # map Y
    mapping = {
        "Adoption": 0,
        "Died": 1,
        "Euthanasia": 2,
        "Return_to_owner": 3,
        "Transfer": 4
    }
    train_Y = train_Y.replace(mapping)

    # compile datasets
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dtest = xgb.DMatrix(test_X)

    # parameters
    param = {
        'objective':            'multi:softprob',
        'bst:max_depth':        6, 
        'bst:eta':              0.05,
        # 'silent':               1, 
        # 'gamma':                0.02,
        # "min_child_weight":     3,
        'num_class':            5,
        # 'verbose':              1,
        # 'subsample':            0.8,
        # 'nthread':              4
    }
    num_rounds = 500

    # train
    print("running the model ...")
    model_xgb = xgb.train(param, dtrain, num_rounds)

    # predict
    print("making prediction ...")
    model_xgb_pred = model_xgb.predict(dtest)

    # submission
    model_xgb_submission = create_submission(model_xgb_pred, test_ID)
    print(model_xgb_submission.head(5))

    # save
    print("saving prediction ...")
    model_xgb_submission.to_csv("submission/model_xgb_train_clean3_1.csv",index=False)


# run if called directly
if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict

from sklearn.neighbors import KNeighborsClassifier

"""
Benchmark classification
------------------------
The parameters are defined according to the results of the OPTUNA trials.
The commented parameters are for the unaugmented data!
The defined parameters are for the augmented data!
"""

# parameters for unaugmented dataset
best_xgb_param_unaug = {'booster': 'dart', 'lambda': 5.0292864803340164e-08, 'alpha': 0.0033512466294373347, 'subsample': 0.7722287536019242, 
                        'colsample_bytree': 0.8788241965652669, 'max_depth': 34, 'min_child_weight': 5, 'eta': 0.05922095844261773, 
                        'gamma': 0.00020027098114354085, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree', 
                        'rate_drop': 4.157639697493719e-07, 'skip_drop': 2.4062928977967765e-06, "verbosity": 0, "objective": "multi:softmax", 
                        "num_class": 30}
best_KNN_param_unaug = {'n_neighbors': 9, 'metric': 'manhattan', 'leaf_size': 46}
best_rf_param_unaug = {'max_depth': 13, 'n_estimators': 132}
best_gb_param_unaug = {'n_estimators': 29, 'learning_rate': 0.016756252304922673, 'max_depth': 11}

# parameters for augmented dataset
best_xgb_param_aug = {'booster': 'gbtree', 'lambda': 2.032151944034843e-06, 'alpha': 9.557956416749613e-09, 'subsample': 0.9511833522668717, 
                      'colsample_bytree': 0.5585628355617734, 'max_depth': 39, 'min_child_weight': 2, 'eta': 0.01731596321744635, 
                      'gamma': 0.013622238453642105, 'grow_policy': 'depthwise, "verbosity": 0, "objective": "multi:softmax", "num_class": 30}
best_KNN_param_aug = {"n_neighbors": 23, "metric": 'euclidean', "leaf_size": 22}
best_rf_param_aug = {"max_depth":15, "n_estimators" : 50}
best_gb_param_aug = {"n_estimators":16, "learning_rate" : 0.02429333140155607, "max_depth" : 10}

def clf_class(X_train, X_test, y_train, y_test, unaugmented=False, xgb_clf=False, KNN_clf=False, rf_clf=False):
    # choose one of these classifiers
    if xgb_clf:
        print("using XGB")
        best_params = best_xgb_param_unaug if unaugmented else best_xgb_param_aug
        clf = XGBClassifier(**best_params)

    elif KNN_clf:
        print("using KNN")
        best_params = best_KNN_param_unaug if unaugmented else best_KNN_param_aug
        clf = KNeighborsClassifier(**best_params)
    elif rf_clf:
        print("using RF")
        best_params = best_rf_param_unaug if unaugmented else best_rf_param_aug
        clf = RandomForestClassifier(**best_params)
    else:
        print("using GB")
        best_params = best_gb_param_unaug if unaugmented else best_gb_param_aug
        clf = GradientBoostingClassifier(**best_params)

    # train classifier
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max) # you can print the classification report for the training set
    # AUC
    y_proba = clf.predict_proba(X_test)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")

    # predict on test set
    if xgb_clf:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)
        test_clf = xgb.train(best_params, dtrain)
        y_pred_val = test_clf.predict(dvalid)

    else :
        y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)
    print(report_test)
    return macro_auc, report_test

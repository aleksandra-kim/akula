import pickle
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, explained_variance_score
from pathlib import Path


def write_pickle(data, filepath):
    """Write ``data`` to a file with .pickle extension"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def read_pickle(filepath):
    """Read ``data`` from a file with .pickle extension"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def read_data(seed, test_size):
    # Read X and Y data
    print("Reading data")
    X1 = read_pickle("X_0_25.pickle")
    X2 = read_pickle("X_25_50.pickle")
    X3 = read_pickle("X_50_75.pickle")
    # X4 = read_pickle("X_75_100.pickle")
    X = np.vstack([X1, X2, X3])
    del X1, X2, X3
    Y = read_pickle("scores.without_lowinf.25000.222201.100000.pickle")
    Y = np.array(Y)[:75000]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=False,
    )
    del X, Y

    return Y_train, Y_test, X_train, X_test


def train_xgboost_model(tag, seed, test_size=0.2):
    """Train gradient boosted tree regressor."""

    fp = Path(f"xgboost.{tag}.pickle")
    Y_train, Y_test, X_train, X_test = read_data(seed, test_size)

    dtrain = xgb.DMatrix(X_train, Y_train)
    dtest = xgb.DMatrix(X_test, Y_test)
    X_dtest = xgb.DMatrix(X_test)

    if fp.exists():
        model = read_pickle(fp)

    else:
        # Define the model
        fp_params = Path(f"xgboost.{tag}.params.json")
        params = dict(
            base_score=np.mean(Y_train),  # the initial prediction score of all instances, global bias
            n_estimators=100,  # number of gradient boosted trees
            max_depth=5,  # maximum tree depth for base learners
            learning_rate=0.3,  # boosting learning rate, xgb's `eta`
            verbosity=3,  # degree of verbosity, valid values are 0 (silent) - 3 (debug)
            booster='gbtree',  # specify which booster to use: gbtree, gblinear or dart
            gamma=0,  # minimum loss reduction to make a further partition on a leaf node of the tree
            subsample=1,  # subsample ratio of the training instance
            colsample_bytree=1,  # subsample ratio of columns when constructing each tree
            reg_alpha=0.1,  # L1 regularization term on weights (xgb’s alpha)
            reg_lambda=0.9,  # L2 regularization term on weights (xgb’s lambda)
            # importance_type="gain",    # for tree models: “gain”, “weight”, “cover”, “total_gain” or “total_cover”
            early_stopping_rounds=30,  # validation metric needs to improve at least once in every early_stopping_rounds
            eval_metric=["rmse"],
            random_state=seed,
            tree_method="hist",
            objective='reg:squarederror',
            min_child_weight=1,
        )
        # Write params into a json file
        with open(fp_params, 'w') as f:
            json.dump(params, f)

        # Train the model
        print("Training the model")
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"], evals=watchlist, verbose_eval=True,
                          early_stopping_rounds=30)
        write_pickle(model, fp)

    # Print results
    y_pred = model.predict(X_dtest)
    r2 = r2_score(Y_test, y_pred)
    explained_variance = explained_variance_score(Y_test, y_pred)
    print("\n===================")
    print(r2, explained_variance)
    print("===================\n")
    return model


def grid_search(tag, seed, test_size=0.2):
    fp = Path(f"grid_search_{tag}.pickle")
    if fp.exists():
        gs = read_pickle(fp)
    else:
        Y_train, Y_test, X_train, X_test = read_data(seed, test_size)
        params = dict(
            base_score=[np.mean(Y_train)],  # the initial prediction score of all instances, global bias
            n_estimators=[100],  # number of gradient boosted trees
            max_depth=[3, 4, 5, 6],  # maximum tree depth for base learners
            learning_rate=[0.3],  # boosting learning rate, xgb's `eta`
            # verbosity=3,  # degree of verbosity, valid values are 0 (silent) - 3 (debug)
            booster=['gbtree'],  # specify which booster to use: gbtree, gblinear or dart
            gamma=[0],  # minimum loss reduction to make a further partition on a leaf node of the tree
            subsample=[0.5, 1.5],  # subsample ratio of the training instance
            colsample_bytree=[0.5, 1.5],  # subsample ratio of columns when constructing each tree
            reg_alpha=[0],  # L1 regularization term on weights (xgb’s alpha)
            reg_lambda=[1],  # L2 regularization term on weights (xgb’s lambda)
            # importance_type="gain",    # for tree models: “gain”, “weight”, “cover”, “total_gain” or “total_cover”
            # early_stopping_rounds=[30],  # validation metric needs to improve at least once in every early_stopping_rounds
            eval_metric=["rmse"],
            random_state=[seed],
            tree_method=["hist"],
            objective=['reg:squarederror'],
            min_child_weight=[1, 300],
        )
        model = xgb.XGBRegressor()
        gs = GridSearchCV(model, params, n_jobs=12,
                          cv=3,
                          scoring='r2',
                          verbose=2, refit=True)

        gs.fit(X_train, Y_train)
        write_pickle(gs, fp)
    return gs


if __name__ == "__main__":
    random_seed = 222201

    # tag_model = "5"
    # model = train_xgboost_model(tag_model, random_seed)

    tag_search = "0"
    search = grid_search(tag_search, random_seed)

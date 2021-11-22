import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from os import path
from copied_project.src.utils import ColumnEncoder
from copied_project.src import DATA_PATH, PROJ_PATH, MODULE_PATH
# Using all_df


df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH, 'processed', f'all_df.csv')))
df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH, 'processed', f'all_df.csv')))

df.position_cleaned.fillna('UNLABELED_POSITION_TITLE', inplace=True)


class SvmEstimator(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # return yhat
        pass

    def transform(self):
        '''I believe this should be a custom transform that applies specific functinality based on the class it belongs to.'''
        pass

class Svm():

    def __init__(self, df):
        self.df = df.copy()# Don't think I actually need this
        self.rslt_df = pd.DataFrame() # This is the RESULT of the TEST set. COULD be called test_df instead
        self.df_wPredictions = pd.DataFrame()
        self.iter_df = pd.DataFrame()
        self.clf = None
        self.iterate_clf = None
        self.n_samples = None
        self.best_tr_tst_dict = dict() # ***THIS IS BASED ON THE INTITIAL RUN*** dict with all of the BEST tr/tst vals and indices, also predictions too
        self.iterate_best_tr_tst_dict = dict()
        self.clf_dictionary = dict()
        self.difference = dict()
        self.unlbl_data = pd.DataFrame()

    # The columns used for model training
    _train_columns = ['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj', 'category']


    def get_train_test_splits(self, df, groups, initial_run, n_splits=4):
        idx = len(df.columns) - 1 # make sure df only has 4 columns
        X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        clf = OneVsRestClassifier(SVC())
        grp_stratified = StratifiedKFold(n_splits, random_state=101)
        grp_stratified.get_n_splits(X, y, groups)
        best_score = 0
        split = 0
        for tr_idx, tst_idx in grp_stratified.split(X, y, groups):
            X_train, X_test = X[tr_idx], X[tst_idx]
            y_train, y_test = y[tr_idx], y[tst_idx]
            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            score = round(accuracy_score(y_test, prediction), 3)
            # Find me the best scoring train set... DO IT NOW!
            if score > best_score:
                best_score = score
                needed_keys =        ['split','score',  'X_train_val','y_train_val','X_train_idx','X_test_val','y_test_val','X_test_idx','prediction']
                corresponding_vals = [split, score,    X_train,       y_train,        tr_idx,     X_test,          y_test,    tst_idx,        prediction]
                if initial_run:
                    self.best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.clf = clf
                else:
                    self.iterate_best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.iterate_clf = clf
            split += 1

    def predict_on_entireDf(self, df, clf):
        """This will allow for manually selecting datasets to iteratively predict and modify existing category values.
            Assumes df.columns = self._train_columns. Otherwise DO NOT include the target variable column!
            """
        if len(df.columns)==4:
            idx = len(df.columns) -1 # index of target variable column, evaluates to: 3
            X = df.iloc[:, 0:idx].to_numpy() # only need the X values for predicting, y is not needed
        else:
            X = df.to_numpy()
        prediction = clf.predict(X)
        df['predicted'] = prediction
        return df
# Above is working on implementing BaseEstimator class
#############################################################

# model = Pipeline([('svm', SVC())])

df_sub = df[['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj', 'category']].copy()
encoder = ColumnEncoder()
df_encode = encoder.column_encoder(df_sub)
df_encode_mini = df_encode.copy()[:50000]


Svm = OneVsRestClassifier(SVC())

X = df_encode_mini[['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj']]
y = df_encode_mini.category

column_trans = make_column_transformer(
    (OneHotEncoder(),['OCC_SERIES', 'COST_CODE']),
    remainder='passthrough')

column_trans.fit_transform(X)

pipe = Pipeline([('transformer', column_trans),
                      ('svm', Svm)])

groups = df_sub.category.unique()
# cross_val_score(pipe, X, y, groups=groups, cv= StratifiedKFold(5), scoring='accuracy')
cross_val_score(pipe, X, y, cv= 5, scoring='accuracy').mean()
# The named steps of the pipeline
# pipe.named_steps['svm']
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random
# Project Imports
# from opr_derog import DATA_PATH
# from opr_derog.src.preprocess_util import dynamic_bar_plot
from copied_project.src import DATA_PATH,MODULE_PATH
from copied_project.src.randomizer import randomizer

# dt_df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH,'raw','dtree.txt'),delimiter='\t'))# This should work, but DATA_PATH won't reload
# dt_df.drop(dt_df.columns[0],axis=1, inplace=True)
# dt_df.rename(columns=dict(zip(dt_df.columns,['position_cleaned','OCC_SERIES','COST_CODE','TIME_ON_JOB','salaryCpiAdj','category'])),inplace=True)
# dt_df.position_cleaned = dt_df.position_cleaned.apply(lambda x: randomizer(x))
# dt_df.category = dt_df.category.apply(lambda x: randomizer(x))
# dt_df.to_csv(path.join(DATA_PATH,'processed','dtree.csv'))


# dt_df = pd.read_csv(path.join(DATA_PATH,'processed','dtree.csv'))
# dt_df.drop(dt_df.columns[0],axis=1, inplace=True)
dt_df = pd.DataFrame(pd.read_csv(path.join(MODULE_PATH,'..','model','df_wPredictions_iterated_predict2021_09_12.csv')))
entire_unlabeled = dt_df.loc[dt_df.category=='unlabeled'.upper()]
all_labeled = dt_df.loc[dt_df.category!='unlabeled'.upper()]


class DecisionTreeHandler():

    def __init__(self,df):
        self.df = df.copy()# Don't think I actually need this
        self.encoded_dict = dict()
        # self.uniform_sample_df = pd.DataFrame() # This is the df we TRAIN on. Superseded by get_train_test_splits
        # self.converse_sample_df = pd.DataFrame() # This is the df we TEST on. Superseded by get_train_test_splits
        self.rslt_df = pd.DataFrame() # This is the RESULT of the TEST set. COULD be called test_df instead
        self.df_wPredictions = pd.DataFrame()
        self.encoded_df = self.column_encoder(df[self._train_columns].copy())
        self.iter_df = pd.DataFrame()
        self.clf = None
        self.iterate_clf = None
        self.n_samples = None
        self.X = self.encoded_df.iloc[:, 0:len(self.encoded_df.columns)-1].to_numpy() # All the X cols/rows
        self.y = self.encoded_df.iloc[:, -1].to_numpy() # All the y rows
        self.best_tr_tst_dict = dict() # ***THIS IS BASED ON THE INTITIAL RUN*** dict with all of the BEST tr/tst vals and indices, also predictions too
        self.iterate_best_tr_tst_dict = dict()
        self.clf_dictionary = dict()
        self.difference = dict()

    # The columns used for model training
    _train_columns = ['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj', 'category']

    def column_encoder(self, df, create_dict=True):
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col].fillna('None', inplace=True)
                le.fit(list(df[col].astype(str).values))
                df[col] = le.transform(list(df[col].astype(str).values))
                # Create dictionary for de-mapping
                if create_dict:
                    self.encoded_dict = {col: dict(zip(le.transform(le.classes_), le.classes_))}
            else:
                df[col].fillna(-999, inplace=True)

        return df

    # def sample_take(self,n_sample,col='category',rs=101):
    #     """take a sample of a dataframe that is a uniformly distributed sample for each value in a given column E.g.:('category') """
    #     category_std = dict(self.encoded_df.groupby(col)['salaryCpiAdj'].std())
    #     category_mean = dict(self.encoded_df.groupby(col)['salaryCpiAdj'].mean())
    #     criteria_df = self.encoded_df[(self.encoded_df.salaryCpiAdj > (self.encoded_df.category.map(category_mean) - self.encoded_df.category.map(category_std))
    #                                    ) & (self.encoded_df.salaryCpiAdj < (self.encoded_df.category.map(category_mean) + self.encoded_df.category.map(category_std)))]
    #
    #     for category in self.encoded_df.category.unique():
    #         self.uniform_sample_df = self.uniform_sample_df.append(criteria_df.loc[criteria_df.category == category].sample(n=n_sample,replace=True,random_state=rs))#replace allows for rplacement if n_samples > category.shape[0]
    #
    #     #also need to generate converse_sample_df here. This df will be comprised of indexes NOT in uniform_sample_df!
    #     self.converse_sample_df = self.encoded_df.copy().loc[~self.encoded_df.index.isin(self.uniform_sample_df.index)]
    #     return self.uniform_sample_df

    # def _shuffle_data(self):
    #     # This 'shuffles' the data. Ensuring more even distribution of categories when splitting the data.
    #     self.encoded_df = self.encoded_df.sample(frac=1, random_state=0)


    def confusion_matrix(self):
        """Plot confusion matrix for outputs. Need to implement log() of sums, otherwise values are too jumbled."""
        mat = confusion_matrix(self.best_tr_tst_dict['y_test_val'], self.best_tr_tst_dict['prediction'] )
        fig,ax = plt.subplots(figsize = (15,15))
        sns.heatmap(mat, square=True, annot=True, cbar=False)
        plt.xlabel('predicted value')
        plt.ylabel('true value')

    # def _merge_preds_selfDf(self, best_dict):
    #     """Introduce predictions from self.rslt_df BACK into self.df.copy() The purpose is to iteratively develop self.df by modifying categories until adequate."""
    #     predictions = pd.DataFrame({'predicted':best_dict['prediction']}, index=best_dict['X_test_idx'])
    #     copy = self.df.copy()
    #     self.df_wPredictions = copy.merge(predictions, how='left', left_index=True, right_index=True) #   left_on=copy.index, right_on=predictions.index, validate='1:1'
    #     self.df_wPredictions.predicted = self.df_wPredictions.predicted.map(self.encoded_dict['category'])
    #     self.df_wPredictions.predicted.fillna('index_not_in_test_set', inplace=True)
    def _merge_preds_selfDf(self, df):
        """Introduce predictions from self.rslt_df BACK into self.df.copy() The purpose is to iteratively develop self.df by modifying categories until adequate."""
        # predictions = pd.DataFrame({'predicted': best_dict['prediction']}, index=best_dict['X_test_idx'])
        copy = self.df.copy()
        for col in df.columns:
            if col in copy.columns:
                copy.drop(columns=col, axis=1, inplace=True)
        self.df_wPredictions = copy.merge(df, how='left', left_index=True,
                                          right_index=True)  # left_on=copy.index, right_on=predictions.index, validate='1:1'
        try:
            # self.df_wPredictions.predicted = self.df_wPredictions.predicted.map(self.encoded_dict['category'])
            self.df_wPredictions.predicted.fillna('IndexNotInTestSet', inplace=True)
        except ValueError:
            print(f'{ValueError}: the column "predicted" is not in the dataframe')

    def _assgn_preds(self, best_dict):
        # This is strictly for the TRAINING set! Introduce predictions array back into df & Map encoded columns back to original values
        idx = best_dict['X_test_idx']
        self.rslt_df = self.df.iloc[idx].copy()
        self.rslt_df['predicted'] = best_dict['prediction']
        self.rslt_df.predicted = self.rslt_df.predicted.map(self.encoded_dict['category'])
        return self.rslt_df

    # Old way that initializes a training set with uniform distributed samples from each df.category. Should not be used unless for testing other functionality
    def _train_test_split(self, tr_df, tst_df):
        """Create your train/test splits. tr_df=uniform_sample_df, tst_df = converse_sample_df"""
        # df and test_df should actually be sub df's from a 'master' df.
        idx = len(tr_df.columns)-1 # makes add/sub columns modular. Make sure y is the last column!
        self.X_train, self.y_train = tr_df.iloc[:, 0:idx].to_numpy(), tr_df.iloc[:, -1].to_numpy()
        self.X_test, self.y_test = tst_df.iloc[:, 0:idx].to_numpy(), tst_df.iloc[:, -1].to_numpy()

    def get_train_test_splits(self, df, groups, initial_run, n_splits=4):
        idx = len(df.columns) - 1 # make sure df only has 4 columns
        X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        clf = tree.DecisionTreeClassifier()
        grp_stratified = StratifiedKFold(n_splits, shuffle=True)
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
        """ ***BASED ON the CURRENT CLF!!!*** This will allow for manually selecting datasets to iteratively predict and modify existing category values .
            The primary use case is for re-classifying 'unlabeled' to an actual category."""
        idx = len(df.columns)-1 # index of target variable column
        X = df.iloc[:, 0:idx].to_numpy() # only need the X values for predicting, y is not needed
        prediction = clf.predict(X)
        df['predicted'] = prediction
        return df

    def set_category_equals_predicted(self, df, iter, category='unlabeled'):
        """The first iteration of predicting on a subset of self.df_wPredictions. Include dict() to record differences in df after category values are updated."""
        pre_ALL_value_counts = df.category.value_counts().to_dict()
        df.loc[df.category == category,'category'] = df.predicted # On these conditions, update category value to predicted
        post_ALL_value_counts = df.category.value_counts().to_dict()

        # Create a dict() to track updated labels
        self.difference[iter] = {}
        for cat in pre_ALL_value_counts.keys():
            diff = post_ALL_value_counts[cat] - pre_ALL_value_counts[cat]
            self.difference[iter][cat] = diff

        print(f'Dataframe has converted {self.difference[iter][category] * -1} "{self.encoded_dict["category"][category]}" values to an ACTUAL category!') # multiply category_reduction by -1 because category_reduction SHOULD be a negative number
        decoded_difference = {self.encoded_dict['category'][k]: v for k, v in self.difference[iter].items()}

    def iterative_predict(self, category=25, num_iters=3): #
        """Once iterative_predict_stepOne has run, run this to iteratively re-assign 'unlabeled' results """

        self.df_wPredictions = self.df_wPredictions[self._train_columns]
        self.df_wPredictions = self.column_encoder(self.df_wPredictions, create_dict=False)
        self.predict_on_entireDf(self.df_wPredictions, clf=self.clf)
        for iter in range(num_iters):
            self.set_category_equals_predicted(self.df_wPredictions, iter, category=category)  # category = 25 = unlabeled
            self.df_wPredictions = self.df_wPredictions[self._train_columns]
            self.get_train_test_splits(self.df_wPredictions, groups= np.array(self.encoded_df.category), initial_run=False)
            print(f'Iteration: {iter} Best split: {self.iterate_best_tr_tst_dict["split"]} {"-" * 5}> Accuracy: {self.iterate_best_tr_tst_dict["score"]}')
            self.predict_on_entireDf(self.df_wPredictions, clf=self.iterate_clf)

        self.set_category_equals_predicted(self.df_wPredictions, iter= num_iters, category=category)
        # self.df.drop(columns='category', axis=1, inplace=True)
        self._merge_preds_selfDf(self.df_wPredictions[['category','predicted']])
        self.df_wPredictions.category = self.df_wPredictions.category.map(self.encoded_dict['category'])
        self.df_wPredictions.predicted = self.df_wPredictions.predicted.map(self.encoded_dict['category'])

    def inital_model_run(self):
        """Train/Tst model & Generate predictions. This is the initial run, I.e. the FIRST run."""
        self.get_train_test_splits(self.encoded_df, groups=np.array(self.encoded_df.category), initial_run=True)
        self._assgn_preds(self.best_tr_tst_dict) # -> creates self.rslt_df
        self._merge_preds_selfDf(pd.DataFrame(self.rslt_df['predicted'], index=self.rslt_df.index)) # takes in a pd.DataFrame as arguement, NOT a pd.Series!
        print(f'Best split: {self.best_tr_tst_dict["split"]} {"-"*5}> Accuracy: {self.best_tr_tst_dict["score"]}')

#############################################
dt = DecisionTreeHandler(all_labeled)
dt.inital_model_run()
labeled_trained = dt.df_wPredictions
labeled_rslt_df = dt.rslt_df
labeledl_discrepancy = labeled_rslt_df[labeled_rslt_df.category!=labeled_rslt_df.predicted]

all_unlabeled = entire_unlabeled.copy()[dt._train_columns]
dt.predict_on_entireDf(all_unlabeled, clf=dt.clf)
all_unlabeled.predicted = all_unlabeled.predicted.map(dt.encoded_dict['category'])
dt.set_category_equals_predicted(all_unlabeled, iter='final',category='UNLABELED')
entire_unlabeled.drop(columns='category', axis=1, inplace=True)
unlabeled_predicted = entire_unlabeled.merge(pd.DataFrame(all_unlabeled[['category','predicted']], index=all_unlabeled.index), left_index=True, right_index=True)
# unlabeled_predicted.predicted.value_counts()
predicted_science = unlabeled_predicted[unlabeled_predicted.predicted=='sciences']
predicted_spec = unlabeled_predicted[unlabeled_predicted.predicted=='misc_specialist'.upper()]

ll = all_labeled.copy()
lp = unlabeled_predicted.copy()
all_have_label = pd.concat([ll,lp])
all_have_label.category.value_counts()

# all_have_label.to_csv(path.join(MODULE_PATH,'..', 'model', f'entire_data_all_label_{today_str}.csv'), index=False)
#############################################


################### Usage ###################

dt = DecisionTreeHandler(dt_df)
# First, run initial_model_run()
dt.inital_model_run()
# After initial, can run iterative_predict() an arbitrary amount of times
dt.iterative_predict(num_iters=10)
# *************** IMPORTANT **********************************
# The pd.DataFrame : dt.df_wPredictions houses the updated category labels
# ************************************************************
df = dt.df_wPredictions

#
# from datetime import date
# today = date.today()
# today_str = today.strftime('%Y_%m_%d')
# df.to_csv(path.join(MODULE_PATH,'..', 'model', f'df_wPredictions_iterated_predict{today_str}.csv'), index=False)
import pandas as pd
#
from os import path
from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OrdinalEncoder
from sklearn import tree
from sklearn.svm import SVC
#
# from nlp_derog import DATA_PATH
from copied_project.src.nlp_transformer import FormatEstimator
# from nlp_derog.src.transformers import (FormatEstimator, FeatureEngTransformer, AssumptionLabelTransformer, DownSamplerTransformer,
# TrainTestSplitOnly)


base_format = FormatEstimator()
df = base_format.df
base_format.fit(df)

# Dataframe, as of now I don't care how I create it- just that we do in fact have it!
df = base_format.fit_transform(df)


class PipelineFeatureEng():
    '''This class instantiates a pipeline to perform feature engineering/extraction as a preliminary step prior to
    introducing any type of modeling. The main purpose of this pipeline is to produce a label for each record in the raw
    data if the record meets explicit conditions from transformers in the pipeline. All data that has been labeled
    is returned.'''

    def __init__(self):
        self.pipe = Pipeline(steps=[
            ('feat_eng', FeatureEngTransformer()) # Engineer our features
            ,('label', AssumptionLabelTransformer()) # Create 'label' column in this step
            ,('sampler', DownSamplerTransformer()) # the fit() method didn't have 'X' param and was making it break- are you freakin serious?? :)
            ,('col_select1', ColumnSelector(['TXTSSN', 'DTMSUBMITSR', 'EFOLDERID', 'TXTBUREAUNAME', 'TXTEMPTYPE',
               'TXTSCI', 'TXTINCIDENTTYPE', 'CURTICKETAMOUNT', 'MEMINCIDENTREPORT',
               'TXTCONOGA_RAW', 'label']))
        ])
        self.label_distribution = self.pipe.named_steps['sampler'].smpl_distr # quick access to 'sampler' pipeline obj


pipe_features = PipelineFeatureEng()
sample = pipe_features.pipe.transform(df)

# Dataframe sample WITH a label
# sample = sample.copy()

train_test = TrainTestSplitOnly()
train_test.get_train_test_splits(sample)
TRAIN_TEST_IDX = train_test.splits_dict

class PipelineAllSteps():
    '''*** NOTE ***
    This class is still in development/progress!

    This class instantiates a pipeline to perform feature engineering and sampling, BUT also includes modeling. This
    is to create single all-purpose pipeline so that all steps can be fed to a GridSearchCV.'''

    def __init__(self):
        self.model = Pipeline(steps=[
            ('feat_eng', FeatureEngTransformer()) # Engineer our features
            ,('label', AssumptionLabelTransformer()) # Create 'label' column in this step

            # The problem with this is, when you change the distribution of the sample, you will ALSO change the shape of the sample,
            # either adding/removing rows. This is an issue because if we fit the data to a gridsearch, we want to specify
            # the indices for our train/test splits. BUT these indices will absolutely change if the shape of our underlying
            # sample changes. We would still want a way to automate different distributions for the sample so we can
            # easily do a grid search.
            ,('sampler', DownSamplerTransformer()) #
            ,('col_select1', ColumnSelector(['TXTINCIDENTTYPE',  'has_fine', 'contains_author_OR_bureau',
                        'pcnt_first_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd']))
            ,(('col_trans', ColumnTransformer([
                    ('scaler', MinMaxScaler(), ['cnt_immediate_rela', 'cnt_frnd']),
                    ('ohe', OneHotEncoder(), ['TXTINCIDENTTYPE', 'has_fine', 'contains_author_OR_bureau'])],
                    remainder='passthrough')))
                ,('estimator', MultinomialNB())
        ])

class PipelineMain():
    '''This class '''
    def __init__(self, estimator=MultinomialNB()):
        # self.num=3
        self.estimator = estimator
        self.best_model = None
        self.best_prediction = None
        self.best_df = pd.DataFrame()
        self.model = Pipeline([
                ('feat_eng', FeatureEngTransformer())
                ,('col_select1', ColumnSelector(['TXTINCIDENTTYPE',  'has_fine', 'contains_author_OR_bureau',
                        'pcnt_first_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd'])) # select only columns that will feed the model
                ,(('col_trans', ColumnTransformer([
                    ('scaler', MinMaxScaler(), ['cnt_immediate_rela', 'cnt_frnd']),
                    ('ohe', OneHotEncoder(), ['TXTINCIDENTTYPE', 'has_fine', 'contains_author_OR_bureau'])],
                    remainder='passthrough')))
                ,('estimator', self.estimator)
            ])

    def iterate_train_test_splits(self, dictlike, df):

        idx = df.shape[1] - 1
        best_score = 0
        for key in dictlike.keys():
            # These are index values
            xtrain = dictlike[key]['train']
            xtest = dictlike[key]['test']
            x_train, y_train = df.iloc[xtrain, 0:idx], df.iloc[xtrain, -1]
            x_test, self.y_test = df.iloc[xtest, 0:idx], df.iloc[xtest, -1]

            self.model.fit(x_train, y_train)
            self.prediction = self.model.predict(x_test)
            score = round(accuracy_score(self.y_test, self.prediction), 3)
            if score > best_score:
                best_score = score
                self.best_model = self.model
                self.best_prediction = self.prediction
                self.best_df = df.iloc[xtest].copy()
                self.best_df['prediction'] = self.best_prediction
            print(f'Split {key} accuracy: {score}')

    def confusion_matrix(self, **kwargs):
        """Plot confusion matrix for outputs. Need to implement log() of sums, otherwise values are too jumbled."""
        # This should be refactored to plot split[i] values from train/test

        mat = confusion_matrix(self.y_test, self.prediction, **kwargs)
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
        plt.xlabel('predicted value')
        plt.ylabel('true value')

    def get_params(self):
        pass

def confusion_matrix_standalone(y_test, prediction, **kwargs):
    """Plot confusion matrix for outputs. Need to implement log() of sums, otherwise values are too jumbled."""
    # This should be refactored to plot split[i] values from train/test

    mat = confusion_matrix(y_test, prediction, **kwargs)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
    plt.xlabel('predicted value')
    plt.ylabel('true value')

pipe = PipelineMain()

# Grid search params, NOTE if we feed several estimators, we cannot add parameters for each estimator too. This is because
# most params in one model are not params for another model!
orig_col_list = ['TXTINCIDENTTYPE', 'has_fine', 'contains_author_OR_bureau',
                                   'pcnt_first_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd']
unkwn_col_list = ['TXTINCIDENTTYPE',  'has_fine', 'contains_author_OR_bureau',
                        'pcnt_first_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd', 'cnt_unknwn_persons']

model_params = {
    # 'estimator__alpha' : [1],
    'col_select1__column_list' : [orig_col_list, unkwn_col_list],
    'col_trans__scaler' : [MinMaxScaler(), Normalizer()],
    'estimator' : [MultinomialNB(),  SVC(), tree.DecisionTreeClassifier(), RandomForestClassifier()] #, tree.DecisionTreeClassifier()
    # 'sampler__' : [{1: .34, 2: .33, 3: .33}, {}]
                }

model = pipe.model
# We are producing an iterable of train/test indicies:
cv = [(TRAIN_TEST_IDX[0]['train'], TRAIN_TEST_IDX[0]['test']), (TRAIN_TEST_IDX[1]['train'], TRAIN_TEST_IDX[1]['test']),
      (TRAIN_TEST_IDX[2]['train'], TRAIN_TEST_IDX[2]['test']), (TRAIN_TEST_IDX[3]['train'], TRAIN_TEST_IDX[3]['test'])]

# Need to fit (xtrain, ytrain)
xtrain, ytrain = sample.iloc[: , 0:10], sample.iloc[: , -1]
grid = GridSearchCV(estimator=model, param_grid=model_params, scoring='f1_micro', cv=cv).fit(xtrain, ytrain) # Add scoring param

# The best scoring pipeline from start to finish:
# best_pipe = grid.best_estimator_

# Put results of different param options in a dataframe, our best model was the RandomForestClassifier w/ MinMaxScaler:
df_results = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')

# Now that we know RFC is the best scoring clf, we will make another Pipeline using this estimator only. HOWEVER, we can
# also just call grid.predict(), which uses the best scoring pipeline to predict!
rfc_model = PipelineMain(estimator=RandomForestClassifier())
rfc_model.iterate_train_test_splits(TRAIN_TEST_IDX, sample)
rfc_model.confusion_matrix()

sample2_df = sample.iloc[TRAIN_TEST_IDX[3]['test']].copy()
sample2_X = sample.iloc[TRAIN_TEST_IDX[3]['test'], :-1].copy()
grid_prediction = grid.predict(sample2_X)
sample2_df['prediction'] = grid_prediction


confusion_matrix_standalone(sample2_df['label'], sample2_df['prediction'])
print(classification_report(sample2_df['label'], sample2_df['prediction']))

# Now see how clf performs on known ABOUT RANDO df: It predicts all as class 1! Very bad
unknwn_idx = [1931, 4601, 6699, 6809, 13154, 16694, 1086, 1133, 1203, 1242, 2311, 4210, 4285, 4315, 4549, 4977, 6123, 6273, 6948, 8596]
rando = df.loc[unknwn_idx].copy()
rando_prediction = grid.predict(rando)

features = FeatureEngTransformer()
rando = features.transform(rando)
rando.columns
# Simple prep of entire data for feeding to model:
df_subset = df.loc[df.TXTINCIDENTTYPE.notna()].copy()
df_subset.reset_index(inplace=True)

# Predicting on all the data:
predictions = rfc_model.best_model.predict(df_subset)
df_subset['prediction'] = predictions
df_subset.prediction.value_counts()

# Organizing our outputs for analysis:
output_about_YOU = df_subset[df_subset.prediction==1]
# output_about_YOU[output_about_YOU.CURTICKETAMOUNT > 0]

output_about_SOMEONE = df_subset[df_subset.prediction==2]

output_about_RANDO = df_subset[df_subset.prediction==3]


best_df = rfc_model.best_df

# pipetest.model[-1].get_params()
# pipetest.model.named_steps['col_trans'].transformers[1]#.get_params()
#
# for steps in pipetest.model.named_steps:
#     for trans in steps.transformers:
#
#         print(steps, trans)

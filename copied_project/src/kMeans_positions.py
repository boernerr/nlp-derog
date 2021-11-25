from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import  check_is_fitted
# from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# from opr_derog.src.visualUtils import array_histplot, dynamic_bar_plot
# from opr_derog import DATA_PATH

from copied_project.src import DATA_PATH, MODULE_PATH
DATA_PATH = path.join(MODULE_PATH,'data')

df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH, 'processed', f'dtree_v2.csv')))
df.position_cleaned.fillna('UNLABELED_POSITION_TITLE', inplace=True)
df_label = df[df.category!='unlabeled'.upper()].copy()


# df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH, 'processed', f'all_df.csv')))
# df.position_cleaned.fillna('UNLABELED_POSITION_TITLE', inplace=True)
# df_label = df[df.category!='unlabeled'].copy()
# df_copy = df_label.copy()


class ScalerTransformer(BaseEstimator, TransformerMixin):
    '''Learning how to implement BaseEstimator and TransformerMixin so that they can be used with custom classes housing both
    a fit() and transform() method.'''

    def __init__(self):
        self._train_columns = ['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj']
        self._y = ['category']
        self.y = None
        self.X = None

    def __str__(self):
        return 'A preliminary transformer performing StandardScaler() functionality.'

    def fit(self, X, y=None):
        '''This can have some basic transformations on X.'''
        # self.X = X[self._train_columns]
        return self

    def transform(self, X):
        '''Transform X by scaling data.'''
        # check_is_fitted(self, 'X') # Need to work on finding appropriate 2nd arg for this func()...
        X_prime = X.copy()
        self.y = X_prime[self._y]
        X_prime = X_prime[self._train_columns]
        # scaler = StandardScaler() # now trying minmax
        scaler = MinMaxScaler()
        scaler.fit(X_prime)
        self.x_prime = scaler.transform(X_prime)
        return self.x_prime
        # return scaler.transform(X_prime)

class KMeansEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, k=25):
        # self.k_means=k_means
        self.k = k
        self.clusterer = KMeans(n_clusters=k, random_state=0)

    def fit(self, X, y=None):
        '''This can have some basic transformations on X. '''
        self.clusterer_ = clone(self.clusterer) # This is following sklearn convention with trailing '_'
        self.clusterer.fit(X)
        return self

    def predict(self, X):
        return self.clusterer.predict(X)

    # def score(self):
    #     '''Is this needed???'''
    #     pass

class Debug(BaseEstimator, TransformerMixin):
    '''Purpose of this class is to create an output step for X. However, this may be unneeded because this step can be done in
    KMeansTransformer.transformer() method.'''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X

########################################################################################################################
# below will be implemented into a class
########################################################################################################################

wss = dict(defaultdict())
silh = dict(defaultdict())

for k in range(20, 35):

    pipe = Pipeline(steps=[('transformer', ScalerTransformer()),
                           ('kmeans', KMeansEstimator(k=k))
                           ])

    pipe.fit(df_copy)

    # Add
    wss[k] = round(pipe['kmeans'].clusterer.inertia_, 3)
    # Silhouette score between transformed X and the 'predicted' (clustered) labels. This has performance issues
    # silh[k] = metrics.silhouette_score(pipe['transformer'].x_prime, pipe['kmeans'].clusterer.labels_)

    predictions = pipe.predict(df_copy)
    df_copy[f'kmeans_{k}'] = predictions
    print(f'model run for {k} clusters')
########################################################################################################################

########################################################################################################################

df_copy.columns
array_histplot(df_copy.OCC_SERIES.value_counts(), bins_=10)
dynamic_bar_plot([[str(i) for i in df_copy.kmeans_15.value_counts().keys()], df_copy.kmeans_15.value_counts().values], log=False, title='Counts per KMeans cluster')
dynamic_bar_plot([wss.keys(), wss.values()], title='KMeans Elbow Plot:  {k| 10<= k <20}')

tstgrp = df_copy[df_copy.kmeans_15==8]
tstgrp2 = df_copy[df_copy.kmeans_15==0]
tstgrp12 = df_copy[df_copy.kmeans_15==5]

tstgrp.salaryCpiAdj.mean()
tstgrp[tstgrp.category=='agent'].salaryCpiAdj.mean()
tstgrp[tstgrp.category=='agent'].AGE.describe()
tstgrp.supervisor.mean()
array_histplot(tstgrp[tstgrp.category=='agent'].salaryCpiAdj, title='Agent grp1')

tstgrp2.salaryCpiAdj.mean()
tstgrp2[tstgrp2.category=='agent'].salaryCpiAdj.mean()
tstgrp2[tstgrp2.category=='agent'].AGE.describe()
tstgrp2.supervisor.mean()
array_histplot(tstgrp2[tstgrp2.category=='agent'].salaryCpiAdj, title='Agent grp2')


tstgrp.category.value_counts()
tstgrp.position_cleaned.value_counts()
tstgrp2.position_cleaned.value_counts()
tstgrp12.position_cleaned.value_counts()
tstgrp12.category.value_counts()
dynamic_bar_plot(tstgrp.COST_CODE.value_counts(), title='grp 8')

silh[k] = metrics.silhouette_score(pipe['transformer'].x_prime, pipe['kmeans'].clusterer.labels_)



df_label['kmeans_predictons'] = predictions
df_label.kmeans_predictons.value_counts(1)

# array_histplot(df_label.category.value_counts(1).values, bins_=15, title='Category distribution as a percentage of total')
# array_histplot(df_label.kmeans_predictons.value_counts(1).values, bins_=np.arange(0,12)/100, title='KMeans predictions distribution as a percentage of total')
# array_histplot(df_label.salaryCpiAdj, bins_=np.arange(0,27)*10000, title='Population Salary')
pos_per_kgrp = df_copy.groupby('kmeans_20').position_cleaned.nunique().sort_values(ascending=False)
cnt_per_kgrp = df_copy.kmeans_20.value_counts(ascending=False)
dynamic_bar_plot([[str(i) for  i in pos_per_kgrp.keys()], pos_per_kgrp.values], title = 'Uniq positions per k-means(20) category')
dynamic_bar_plot([[str(i) for  i in cnt_per_kgrp.keys()], cnt_per_kgrp.values], title = 'Count positions per k-means(20) category')


grp6 = df_copy[df_copy.kmeans_20==6]
grp6.category.value_counts()
grp6.position_cleaned.value_counts()
grp6test = grp6[grp6.position_cleaned =='SUPVY SPECIAL AGENT']
grp6test.WORKING_TITLE.value_counts()
grp6.salaryCpiAdj.mean()



grp1 = df_label[df_label.kmeans_predictons==1]
grp1.category.value_counts()
grp1.position_cleaned.value_counts()
grp1.category.value_counts()
grp1test = grp1[grp1.position_cleaned =='SUPVY SPECIAL AGENT']
grp1test.WORKING_TITLE.value_counts()
grp1.salaryCpiAdj.mean()

# Group 8
group8 = df_label[df_label.kmeans_predictons==8]
group8.position_cleaned.value_counts()
group8.category.value_counts()
group8_eng = group8[group8.category =='']
group8.salaryCpiAdj.describe()
array_histplot(group8.salaryCpiAdj,  title='Grp 8 salaryCpiAdj')

# Group 11
group11 = df_label[df_label.kmeans_predictons==11]
group11.position_cleaned.value_counts().keys().nunique()
group11.category.value_counts()
group11.salaryCpiAdj.mean()
group11_test = group11[group11.category =='misc_specialist']
array_histplot(group11.COST_CODE,  title='Grp 11 Population Salary')

# Group 12
group12 = df_label[df_label.kmeans_predictons==12]
group12.position_cleaned.value_counts()
group12.category.value_counts()
grp12_tst = group12[group12.salaryCpiAdj< 40000]
group12.salaryCpiAdj.describe()
array_histplot(group12.salaryCpiAdj,  title='Grp 12 salaryCpiAdj')

# Group 12
group16 = df_label[df_label.kmeans_predictons==16]
group16.position_cleaned.value_counts()
group16.category.value_counts()
group16tst = group16[group16.category =='engineer']
group16.salaryCpiAdj.mean()
array_histplot(group16.OCC_SERIES,  title='Grp 12 COST_CODE')

# Group 21
group21 = df_label[df_label.kmeans_predictons==21]
group21.position_cleaned.value_counts().keys().nunique()
group21.category.value_counts()
group21.salaryCpiAdj.mean()
group21 = group21[group21.category =='misc_specialist']
array_histplot(group21.COST_CODE,  title='Grp 21 Population Salary')

# Group 23
group23 = df_label[df_label.kmeans_predictons==23]
group23.position_cleaned.value_counts()
group23.category.value_counts()
group23.salaryCpiAdj.mean()
group23_test = group23[group23.category =='misc_specialist']



df_label[df_label.position_cleaned=='MECHANICAL ENGINEER'].kmeans_predictons.value_counts()

k25_group_names = {12: 'labor_loPaid',
                   8: 'labor_hiPaid',
                   23: 'senior_management'}


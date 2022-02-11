import string
import re
import itertools
import os
import pandas as pd
import unicodedata
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.cluster import KMeansClusterer
# from nltk.corpus import wordnet as wn  # This doesn't work on FBINet
from nltk import sent_tokenize
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from copied_project.src import DATA_PATH

# dictionary to determine voice of PRON's in MEMINCIDENTREPORT
PRON_DICT = {'first': ['I', 'i', 'me', 'my', 'mine', 'myself'],
               # 'second': [], # don't think we'll need this
             'third': ['he', 'him', 'his', 'she', 'her', 'hers', 'their', 'their', 'they', 'them'],
             # Want to couple any possessive PRON with any noun in RELATIVE_DICT and search for 'my {noun}'. I'm assuming
             # this should be a good indication of determining if incident is about SOMEONE YOU KNOW
             'possessive': ['my']

}

# dictionary to determine instances of relations mentioned in MEMINCIDENTREPORT
RELATIVE_DICT = {
    'immediate': ['mom', 'mother', 'dad', 'father', 'brother', 'sister', 'son', 'daughter', 'grandmother', 'grandfather',
                  'grandma', 'grandpa', 'parents', 'family', 'husband', 'wife', 'spouse'],
    'distant': ['aunt', 'uncle', 'cousin', 'nephew', 'niece'],
    'friend': ['friend', 'friends', 'boyfriend', 'girlfriend', 'boy friend', 'girl friend']
}

BUREAU_IDENTIFIERS = ['employee', 'writer']

UNKWN_IDENTIFIERS = ['man', 'woman', 'person', 'boy', 'girl', 'individual', 'individuals'] # The ordering matters. Bc regex isn't applying the pattern to the last item in the list.

# Transformer Classes
class FormatEstimator(BaseEstimator, TransformerMixin):
    '''Class for reading in .csv file and doing VERY BASIC cleanup to include:
    - Dropping last row
    - Dropping duplicate rows
    - Subsetting the data to only include non NULL MEMINCIDENTREPORT rows'''

    def __init__(self, path_ = os.path.join(DATA_PATH, 'raw', 'SRSP_SELF_REPORT__lawEnforceInteraction.csv')):
        self.file_path = path_
        self.df = pd.DataFrame()
        self._create_df()

    def fit(self, X):
        return self

    def transform(self, X):
        '''Exclude last row, convert DTMSUBMITSR to date dtype.'''
        x_prime = X.copy()

        x_prime = X.iloc[:-1]
        x_prime = x_prime[x_prime.MEMINCIDENTREPORT.notna()]
        x_prime.DTMSUBMITSR = pd.to_datetime(x_prime.DTMSUBMITSR)
        x_prime.drop_duplicates(inplace=True)
        return x_prime

    def _create_df(self):
        df = pd.read_csv(self.file_path, engine='python')
        self.df = df

class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    '''Transformer for:
    - Normalizing MEMINCIDENTREPORT column
    - Computing percent of pronouns used in MEMINCIDENTREPORT that are 1st/3rd person.'''

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        '''Create columns for:
        - Count of occurrences for:
            - 1st person pronoun and 3rd person PRON
            - Immmediate and distant relatives
            - friends
            - Unknown persons indicated
        - Ratio of 1st : 3rd person PRON and possessive PRON in conjunction with relatives *not completely accurate bc using 'my'*
        - Flag if text contains reference to the author or other FBI persons
        - Flag if record has a fine/monetary penalty

        We would assume a MEMINCIDENTREPORT with a ratio of more 1st PRON than 3rd PRON would heavily indicate the MEMINCIDENTREPORT
        is a description of the author.
        '''

        x_prime = X.copy()
        x_prime.MEMINCIDENTREPORT = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._normalize_text(x_))

        x_prime['contains_bureau_persons'] = x_prime.MEMINCIDENTREPORT.str.contains('|'.join(i for i in BUREAU_IDENTIFIERS)) # FBI flag; bool
        x_prime['cnt_unknwn_persons'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, UNKWN_IDENTIFIERS, regex=r'[^\w]| ', substr_flag=False))
        x_prime['last_name'] = x_prime.TXTBUREAUNAME.astype(str).apply(lambda x_: self._find_lastname(x_))
        x_prime['contains_author'] = x_prime.apply(lambda x: x.last_name in x.MEMINCIDENTREPORT, axis=1)#.map({True: 1, False: 0}) # last name flag; bool

        # Combine contains_bureau_persons flag with self author flag
        x_prime['contains_author_OR_bureau'] = x_prime.apply(lambda x:  x.contains_bureau_persons > 0 or x.contains_author > 0 , axis=1) # flag combining contains_author | contains_bureau_persons
        x_prime['possessive_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.possessive_entity_checker(x_, regex=r'[^\w]|'))
        x_prime['cnt_third_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, PRON_DICT['third'], regex=r'[^\w]| ', substr_flag=True)) # Notice there is NO space after '|' - this is important!
        x_prime['cnt_first_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, PRON_DICT['first'], regex=r'[^\w]|'))
        x_prime['cnt_immediate_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['immediate'], regex=r'[^\w]| ')) # Notice there IS a space after '|' - this is important!
        x_prime['cnt_dstnt_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['distant'], regex=r'[^\w]| '))
        x_prime['cnt_frnd'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self.entity_checker(x_, RELATIVE_DICT['friend'], regex=r'[^\w]| '))
        x_prime['has_fine'] = x_prime.CURTICKETAMOUNT.apply(lambda x_: 0 if not x_ else 1)

        # Percentages of entities mentioned
        x_prime['pcnt_poss_rela'] = (x_prime.possessive_rela / x_prime.cnt_first_pron).fillna(0) # The percent of possessive relatives ('my ...'). This should highlight records about someone YOU KNOW

        ttl_pron = x_prime.cnt_third_pron + x_prime.cnt_first_pron
        x_prime['pcnt_first_pron'] = round(x_prime.cnt_first_pron/ ttl_pron, 2)
        x_prime.pcnt_first_pron.fillna(0, inplace=True)
        x_prime['pcnt_third_pron'] = round(x_prime.cnt_third_pron / ttl_pron, 2)
        x_prime.pcnt_third_pron.fillna(0, inplace=True)

        x_prime = x_prime[['TXTINCIDENTTYPE', 'MEMINCIDENTREPORT', 'has_fine', 'contains_author_OR_bureau',
                           'contains_bureau_persons', 'contains_author', 'cnt_unknwn_persons', 'pcnt_first_pron', 'cnt_first_pron',
                           'cnt_third_pron', 'pcnt_poss_rela', 'cnt_immediate_rela', 'cnt_frnd']]
        x_prime = x_prime[x_prime.TXTINCIDENTTYPE.notna()] # THis doesn't remove the np.nan in TXTINCIDENTTYPE
        return x_prime

    def _normalize_text(self, str_):
        '''Tokenize words and convert to lowercase. As of now, the tokenized column is converted back into a string in
        all lowercase.'''
        tokenized = [word.lower() for word in wordpunct_tokenize(str_)]
        return ' '.join(i for i in tokenized)
        # return [word.lower() for word in wordpunct_tokenize(str_)]

    def entity_checker(self, str_, srch_critria, regex=r'[^\w]| ', substr_flag=False):
        '''Instead of individual functions for each entity, this is one function that will search a string for srch_critria
        given as one of the arguments.

        Param
        -----
        str_ : string
            The string that is being searched. i.e. text field in dataframe.
        srch_critria : list
            list of strings to search in str_. Put another way, returns True if srch_critria in str_
        regex : string
            The regex pattern used in conjuction with srch_critria. This will most likely remain unchanged from default.
        substr_flag : bool
            This flags whether srch_critria can be found as a substring of a larger word within str_ i.e. if True,
            searching for the word 'he' against a string containing the word 'sheriff' WILL return an occurrence because
            'he' is in s(he)riff.
        '''

        if not substr_flag:
            words = [' ?' + i for i in srch_critria]
        # The below regex prevents 'sheriff' from returning True when matching against pronouns. Else, 's(he)riff'
        # would match 'he'
        else:
            words = [' ?[^a-z]' + i for i in srch_critria]
        reg = regex.join(i for i in words)
        return len(re.findall(reg, str_) )

    def possessive_entity_checker(self, str_, srch_critria=None, regex=r'[^\w]|'):
        '''Check for possessive ('my') use of relatives; i.e. 'my uncle'. This is one function that will search a string for srch_critria
        given as one of the arguments.

        Param
        -----
        str_ : string
            The string that is being searched. i.e. text field in dataframe.
        srch_critria : list
            list of strings to search in str_.
        regex : string
            The regex pattern used in conjuction with srch_critria.
        '''

        if srch_critria:
            words = [' ?my' + i for i in srch_critria]
        else:
            words = [' ?my ' + item for sublist in [ i for i in [RELATIVE_DICT[i] for i  in RELATIVE_DICT.keys()]] for item in sublist]

        self.reg = regex.join(i for i in words)
        # print(re.findall(self.reg, str_)) # for testing ONLY
        return len(re.findall(self.reg, str_) )

    def _find_lastname(self, str_, sep=','):
        '''Return just the last name from the TXTBUREAUNAME field. Delimited by a comma (','). This could be initially
        included in the source data by changing the underlying SQL query.

        Params
        ------
        str_ : string
            This indicates the string that sep is searched in.
        sep : string
            This is the string character we are searching for i.e. if sep=',' then we are searching str_ for the position
            of sep.
        '''

        comma_idx = str_.find(sep)
        return str_[: comma_idx].lower()

class TextNormalizer(BaseEstimator,TransformerMixin):

    def __init__(self, text_col):
        self.language = 'english'
        self.stopwords = set(nltk.corpus.stopwords.words(self.language))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.text_col = text_col

    def is_punct(self,token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self,token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        """
        Normalize the text by lemmatization by removing stopwords and punct.
        This function should be implemented by:
        pd.Series.apply(lambda x: normalize(x))

        Param
        -----
        document : pd.Series
            In this case, we are applying normalize() to a pd.Series in a pd.DataFrame. Each row, then, is a distinct 'document'.
            The pd.Series in question should be one long string.

        """
        doc = pos_tag(wordpunct_tokenize(document))
        # j[0] = token, j[1] = tag
        return [self.lemmatize(j[0].lower(), j[1]) for j in [k for k in doc]
                if not self.is_punct(j[0]) and not self.is_stopword(j[0])]

    def lemmatize(self, token, pos_tag):
        """
        Maps nltk.pos_tag to WordNet tag equivalent.
        Assumes a (token, pos_tag) tuple as input.

        Param
        -----
        token : str
            A token, i.e. a word
        pos_tag : str
            nltk PartOfSpeech tag
        """
        tag = {
             'N': wn.NOUN,
             'V': wn.VERB,
             'R': wn.ADV,
             'J': wn.ADJ
         }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        """Generic fit func() to comply with SKLEARN."""
        return self

    def transform(self, X):
        """Modify the specified text_col by normalizing it. Then convert it from a list to one big string."""

        X[self.text_col] = X[self.text_col].apply(lambda x: self.normalize(x))
        X[self.text_col] = X[self.text_col].apply(lambda x: ' '.join(i for i in x))
        return X

class AssumptionLabelTransformer(BaseEstimator, TransformerMixin):

    CODIFY_DICT = {
        1: 'regarding_self',
        2: 'regarding_knwn_prsn',
        3: 'regarding_unkn_prsn'
    }

    def __init__(self, text_col=None):
        self.text_col = text_col
        self.unlbl_idx = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''We will create an X and y in this function. ACTUALLY we will NOT create y in this transformer'''

        # This assumption holds almost ALL the time; Meaning if there is a ticket $ amount, the incident is about the AUTHOR!
        # Only exception found: X.loc[17684]
        # X.loc[X.CURTICKETAMOUNT > 0, 'label'] = 1 # This was old criteria for ABOUT YOURSELF
        X.loc[(X.has_fine == 1) | (X.contains_bureau_persons > 0) | (X.contains_author > 0) | (X.pcnt_first_pron == 0), 'label'] = 1
        X.loc[(X.has_fine == 0) & (X.pcnt_first_pron < .8) & (X.pcnt_poss_rela >= .2) & ((X.cnt_immediate_rela > 0) | (X.cnt_frnd > 0)), 'label'] = 2
        # This is old criteria, but doesn't return as many results as it previously did. THis is because I changed the logic for counting first_pron
        # X.loc[(X.has_fine == 0) & (X.pcnt_first_pron < .30) & (X.pcnt_poss_rela >= .5) & ((X.cnt_immediate_rela > 0 ) | (X.cnt_frnd > 0 )), 'label'] = 2
        X.loc[(X.cnt_unknwn_persons > 4) & (X.contains_author == 0) & (X.pcnt_poss_rela > 0), 'label'] = 3 # Changed X.cnt_unknwn_persons > 2 to > 4
        self.unlbl_idx = X[X.label.isna()].index

        return X[X.label.notna()]

class OneHotVectorizer(BaseEstimator, TransformerMixin):
    '''Class for one-hotting words(i.e. features) into an array in order to feed to a model downstream.'''

    def __init__(self, text_col):
        self.vectorizer = CountVectorizer(input='content', decode_error='ignore', binary=True)
        self.text_col = text_col

    def fit(self, X, y=None):
        '''Generic fit function.'''

        return self

    def transform(self, X):
        '''Vectorize self.text_col into an array of arrays. This is so we can feed the AoA to a model. '''

        freqs = self.vectorizer.fit_transform(X[self.text_col])
        return [freq.toarray()[0] for freq in freqs]

class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='most_frequent'):
        self._col_names = None
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

    def fit(self, X, y=None):
        self._col_names = X.columns
        self.imputer.fit(X)

    def transform(self, X):
        X_ = self.imputer.transform(X)
        X_ = pd.DataFrame(X_, columns = self._col_names)
        return X_

class DownSamplerTransformer(BaseEstimator, TransformerMixin):
    '''Once we have labeled data, this class will allow us to select the proportion of each label to be included in a
    train set. This is necessary because we want to approximate the distribution of the population in the sample used
    for training.
    '''

    def __init__(self):
        self.baseline_weights = {1: .60, 2: .30, 3: .10}

    def fit(self):
        return self

    def transform(self, X, colmn='label', smpl_distr=None, rand=96):
        '''Set the distribution of a training set so that the least frequent labeled class' proportion is set first.
        Note: This will break IF the percentages present in smpl_distr.values() for a given class label evaluates to a
        number higher than the amount of records for that class in X. To fix, this we could simply allow sampling WITH
        replacement.

        Params
        ------
        X : pd.Dataframe
            pretty self explanatory, but should include a column with labels
        colmn : str
            a column in X that houses the labels for each row
        dictLike : dict()
            a dictionary that should have class labels as keys and percentages as values i.e.
            The percentages should add up to 1

        Ideal baseline distribution for smpl_distr:
        {
        1: .50, # labeled as about YOURSELF
        2: .40, # labeled as about SOMEONE YOU KNOW
        3: .10} # labeled as about RANDO
        '''
        if not smpl_distr:
            smpl_distr = self.baseline_weights
        xprime = pd.DataFrame()
        labels_cnts = X[colmn].value_counts().to_dict()
        # The labeled class with the lowest count
        least_value = min(labels_cnts, key=labels_cnts.get)
        # This makes use of ALL of the records of the smallest labeled class by setting the count of this class' records to
        # the proportion of the class in smpl_distr. The shape[0] of xprime is set based on least_value/ the value of the key for
        # least value in smpl_distr
        try:
            df_shape0 = round(labels_cnts[least_value] / smpl_distr[least_value], 0)
        except KeyError as e:
            print(f'KeyError of {e}: The dictLike object key(s) do not match the values within colmn. Make sure the value '
                  f'of {e} is present in the keys of dictLike argument!')

        for label in labels_cnts.keys():
            sample_amt = int(round(df_shape0 * smpl_distr[label], 0))
            print(f'Class label: {label} results in {sample_amt} rows')
            xprime = xprime.append(X[X[colmn] == label].sample(n=sample_amt, random_state=rand))

        return xprime


# Estimator Classes
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

    def transform(self, X):
        '''let's try return self to see if this gets rid of the AttributeError.'''
        return self

    def predict(self, X):
        return self.clusterer.predict(X)

    # def score(self):
    #     '''Is this needed???'''
    #     pass

class KMeansClusters(BaseEstimator, TransformerMixin):
    '''Directly from the book, this KMeans instance is from nltk instead of directly from sklearn.'''

    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """Fits k-means to one-ht encoded vectorize documents."""
        return self.model.cluster(documents, assign_clusters=True)

class HierarchicalClusters( BaseEstimator, TransformerMixin):

    def __init__(self):
        self.model = AgglomerativeClustering()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''This is an agglomerative model. Briefly, agglomerative models work from the 'bottom up'. Where every point is
         initially it's own cluster BUT is iteratively merged together up the hierarchy.'''

        clusters = self.model.fit_predict(X)
        self.y = self.model.labels_
        self.children = self.model.children_

        return clusters

class NaiveBayesClf (BaseEstimator, TransformerMixin):

    def __int__(self):
        self.model = MultinomialNB()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        xprime = X

class TrainTestSplitOnly():
    '''Class with sole purpose of storing train/test indices within a dictionary. Implementation below which can be
    copied/pasted in your IDE:

        train_test = TrainTestSplitOnly()
        # If df is pd.DataFrame: make sure to use .iloc as shown below
        train_test.get_train_test_splits(df, df.iloc[:, -1], n_splits=4)
        # Access the indices of the splits by entering key = {'test', 'train'}; update i to an integer: {0 <= i < n_splits}
        split_dict = train_test.splits_dict[i]['test']
        # Check that the splits DO in fact evenly split the data by the supplied label
        df_labeled.iloc[testing_split_idx].label.value_counts()

    '''
    def __init__(self):
        self.splits_dict = defaultdict(lambda: defaultdict())

    def get_train_test_splits(self, df, groups, n_splits=4):

        if isinstance(df, pd.DataFrame):
            idx = len(df.columns) - 1
            X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        # Else clause handles if df is an np.array or matrix
        else:
            idx = df.shape[1] - 1
            X, y = df[:, 0:idx], df[:, -1]

        grp_stratified = StratifiedKFold(n_splits, random_state=101)
        grp_stratified.get_n_splits(X, y, groups)

        for i, (tr_idx, tst_idx) in enumerate(grp_stratified.split(X, y, groups=y)):
            # Below 2 lines not needed because we ONLY want the indices of train and test
            # X_train, X_test = X[tr_idx], X[tst_idx]
            # y_train, y_test = y[tr_idx], y[tst_idx]
            self.splits_dict[i] = {'train': tr_idx, 'test': tst_idx}

class TrainTestSplitWrapper():

    def __init__(self, split_type=StratifiedKFold()):
        self.split_type = split_type
        self.clf = None
        # Add a pipeline here?
        self.model = Pipeline([])

    def confusion_matrix(self, **kwargs):
        """Plot confusion matrix for outputs. Need to implement log() of sums, otherwise values are too jumbled."""
        mat = confusion_matrix(self.best_tr_tst_dict['y_test_val'], self.best_tr_tst_dict['prediction'], **kwargs)
        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
        plt.xlabel('predicted value')
        plt.ylabel('true value')

    def get_train_test_splits(self, df, groups, initial_run=True, n_splits=4):

        # Think I actually may need the isinstance() method BECAUSE if dataframe: I have to use to_numpy(); else: already np.array
        if isinstance(df, pd.DataFrame):
            idx = len(df.columns) - 1
            X, y = df.iloc[:, 0:idx].to_numpy(), df.iloc[:, -1].to_numpy()
        # Else clause handles if df is an np.array or matrix
        else:
            idx = df.shape[1] - 1
            X, y = df[:, 0:idx], df[:, -1]
        # clf = MultinomialNB()
        clf = tree.DecisionTreeClassifier()
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
                needed_keys = ['split', 'score', 'X_train_val', 'y_train_val', 'X_train_idx', 'X_test_val', 'y_test_val',
                               'X_test_idx', 'prediction']
                corresponding_vals = [split, score, X_train, y_train, tr_idx, X_test, y_test, tst_idx, prediction]
                if initial_run:
                    self.best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.clf = clf
                else:
                    self.iterate_best_tr_tst_dict = dict(zip(needed_keys, corresponding_vals))
                    self.iterate_clf = clf
            split += 1


class EntityPairs(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(EntityPairs, self).__init__()

    def pairs(self, document):
        return list(itertools.permutations(set(document), 2))

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return [self.pairs(document) for document in documents]



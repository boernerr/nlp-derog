import string
import re
import os
import pandas as pd
import unicodedata
# import textblob

import nltk
import nltk.cluster.util
from collections import defaultdict
from nltk import pos_tag, wordpunct_tokenize, CFG
from nltk.cluster import KMeansClusterer
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize



from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from copied_project.src import DATA_PATH
from copied_project.src._util_plot_funcs import plot_dendrogram
from copied_project.src import k_means_nlp
from copied_project.src.kMeans_positions import KMeansEstimator
# from copied_project.src import text_normalizer
# from importlib import reload
# reload(text_normalizer)

# dictionary to determine voice of PRON's in MEMINCIDENTREPORT
PRON_DICT = {'first': ['I', 'i', 'me', 'my', 'mine','myself'],
               # 'second': [], # don't think we'll need this
               'third': ['he', 'him', 'she', 'hers', 'their', 'their', 'they', 'them']
}

# dictionary to determine instances of relations mentioned in MEMINCIDENTREPORT
RELATIVE_DICT = {
    'immediate': ['mom', 'mother', 'dad', 'father', 'brother', 'sister', 'son', 'daughter', 'grandmother', 'grandfather',
                  'grandma', 'grandpa', 'parents', 'family'],
    'distant': ['aunt', 'uncle', 'cousin', 'nephew', 'niece'],
    'friend': ['friend', 'friends', 'boyfriend', 'girlfriend' 'boy friend', 'girl friend']
}

class FormatEstimator(BaseEstimator, TransformerMixin):
    '''Class for reading in .csv file and doing VERY BASIC cleanup to include:
    - Dropping last row
    - Dropping duplicate rows
    - Subsetting the data to only include non NULL MEMINCIDENTREPORT rows'''

    def __init__(self, file_path = os.path.join(DATA_PATH, 'raw', 'complaints-2021-08-30_12_09.csv')):
        self.file_path = file_path
        self.df = pd.DataFrame()
        self._create_df()

    def fit(self, X):
        return self

    def transform(self, X):
        '''Exclude last row, convert DTMSUBMITSR to date dtype.'''
        x_prime = X.copy()
        x_prime = self.convert_to_underscore(x_prime)
        x_prime = x_prime.loc[x_prime.consumer_complaint_narrative.notna()][['complaint_id', 'date_received','issue', 'sub_issue', 'consumer_complaint_narrative', 'state']]
        # x_prime = X.iloc[:-1]
        # x_prime = x_prime[x_prime.MEMINCIDENTREPORT.notna()]
        # x_prime.DTMSUBMITSR = pd.to_datetime(x_prime.DTMSUBMITSR)
        # x_prime.drop_duplicates(inplace=True)

        return x_prime

    def convert_to_underscore(self, df):
        '''Convert column names to snake_case.'''
        newList = []
        for string in df.columns:
            # Convert spaces(' ') and dashes('-') to '_'
            newList.append(re.sub(' |-', '_', string).lower())
        df.rename(columns=dict(zip(df.columns, newList)), inplace=True)
        return df

    def _create_df(self):
        df = pd.read_csv(self.file_path, engine='python')
        self.df = df

class FeatureEngTransformer(BaseEstimator, TransformerMixin):
    '''Transformer for:
    - Normalizing MEMINCIDENTREPORT column
    - Computing percent of pronouns used in MEMINCIDENTREPORT that are 1st/3rd person.'''

    def __init__(self):
        self.name = 'feature_engineer'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''Create columns for count of occurrences of 1st person pronoun and 3rd person pronouns.
        We would assume a MEMINCIDENTREPORT with a ratio of more 1st PRON than 3rd PRON would heavily indicate the MEMINCIDENTREPORT
        is a description of the author.
        That leaves us '''

        x_prime = X.copy()
        # x_prime['normalized'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._normalize_text(x_))

        x_prime['cnt_third_pron'] = x_prime.consumer_complaint_narrative.apply(lambda x_: self._is_third_pers_PRN(x_))
        x_prime['cnt_first_pron'] = x_prime.consumer_complaint_narrative.apply(lambda x_: self._is_first_pers_PRN(x_))
        x_prime['cnt_immediate_rela'] = x_prime.consumer_complaint_narrative.apply(lambda x_: self._is_relative_immediate(x_))
        x_prime['cnt_dstnt_rela'] = x_prime.consumer_complaint_narrative.apply(lambda x_: self._is_relative_dist(x_))
        x_prime['cnt_frnd'] = x_prime.consumer_complaint_narrative.apply(lambda x_: self._is_friend(x_))
        # x_prime['has_fine'] = x_prime.consumer_complaint_narrative.apply(lambda x_: 1 if not x_ else 0)

        ttl_pron = x_prime.cnt_third_pron + x_prime.cnt_first_pron
        x_prime['pcnt_first_pron'] = round(x_prime.cnt_first_pron/ ttl_pron, 2)
        x_prime.pcnt_first_pron.fillna(0, inplace=True)
        # x_prime['pcnt_first_pron'] = self._convert_denom_zero(x_prime.cnt_first_pron, ttl_pron)
        x_prime['pcnt_third_pron'] = round(x_prime.cnt_third_pron / ttl_pron, 2)
        x_prime.pcnt_third_pron.fillna(0, inplace=True)
        # x_prime['pcnt_third_pron'] = self._convert_denom_zero(x_prime.cnt_third_pron, ttl_pron)
        # x_prime['tokens'] = x_prime.normalized.apply(lambda x_: self._normalize_text(x_)) # Not necessary for now

        return x_prime

    # def _convert_denom_zero(self, a, b):
    #     return a/ b if b else 0

    def _normalize_text(self, str_):
        '''Tokenize words and convert to lowercase. As of now, the tokenized column is converted back into a string in
        all lowercase.'''
        tokenized = [word.lower() for word in wordpunct_tokenize(str_)]
        return ' '.join(i for i in tokenized)
        # return [word.lower() for word in wordpunct_tokenize(str_)]

    def _is_relative_immediate(self, str_):
        '''Check if any immediate relatives are mentioned in text.
        Example REGEX pattern for matching the word 'son' should be: ' ?son[^\w]'

        Translation: find all words in words variable that have 0 or 1 whitespaces before the word or any char after the
        word that is NOT [a-z0-9]. i.e. any char that IS either whitespace or punctuation.
        '''

        words = RELATIVE_DICT['immediate']
        words = [' ?' + i for i in words]
        regex = r'[^\w]| '.join(i for i in words)
        # instances = re.findall(regex, str_)
        return len(re.findall(regex, str_) )

    def _is_relative_dist(self, str_):
        '''Check if any distant relatives are mentioned in text.
        Example REGEX pattern for matching the word 'aunt' should be: ' ?aunt[^\w]'

        Translation: find all words in words variable that have 0 or 1 whitespaces before the word or any char after the
        word that is NOT [a-z0-9]. i.e. any char that IS either whitespace or punctuation.
        '''

        words = RELATIVE_DICT['distant']
        words = [' ?' + i for i in words]
        regex = r'[^\w]| '.join(i for i in words)
        # instances = re.findall(regex, str_)
        return len(re.findall(regex, str_))

    def _is_friend(self, str_):
        '''Check if any friends are mentioned in text.
        Example REGEX pattern for matching the word 'friend' should be: ' ?friend[^\w]'

        Translation: find all words in words variable that have 0 or 1 whitespaces before the word or any char after the
        word that is NOT [a-z0-9]. i.e. any char that IS either whitespace or punctuation.
        '''

        words = RELATIVE_DICT['friend']
        words = [' ?' + i for i in words]
        regex = r'[^\w]| '.join(i for i in words)
        # instances = re.findall(regex, str_)
        return len(re.findall(regex, str_))


    def _is_first_pers_PRN(self, str_):
            '''Check if 1st person pronouns are in text.
            Example REGEX pattern for matching the word 'I' should be: ' ?I[^\w]'

            Translation: find first person PRON that have 0 or 1 whitespaces before the PRON or any char after the PRON that
            is NOT [a-z0-9]. i.e. any char that IS either whitespace or punctuation.
            '''

            prons = PRON_DICT['first']
            prons = [' ?' + i for i in prons]
            regex = r'[^\w]| '.join(i for i in prons)
            # instances = re.findall(regex, str_)
            return len(re.findall(regex, str_) )

    def _is_third_pers_PRN(self, str_):
        '''Check if 3rd person pronouns are in text.
        Example REGEX pattern for matching the word 'he' should be: ' ?he[^\w]'

        Translation: find third person PRON that have 0 or 1 whitespaces before the PRON or any char after the PRON that
        is NOT [a-z0-9]. i.e. any char that IS either whitespace or punctuation.
        '''

        prons = PRON_DICT['third']
        prons = [' ?' + i for i in prons]
        regex = r'[^\w]| '.join(i for i in prons)
        # instances = re.findall(regex, str_)
        return len(re.findall(regex, str_) )

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

class TextCountVectorizer(BaseEstimator, TransformerMixin):

     def __init__(self, text_col):
         self.text_col = text_col

     def fit(self, X):
        return self

     def transform(self, X):
        vectorizer = CountVectorizer(input='content')
        X = vectorizer.fit_transform(X[self.text_col])
        self.feature_names = vectorizer.get_feature_names()
        return X



class AssumptionLabelTransformer(BaseEstimator, TransformerMixin):

    CODIFY_DICT = {
        1: 'regarding_self',
        2: 'regarding_knwn_prsn',
        3: 'regarding_unkn_prsn'
    }

    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        '''We will create an X and y in this function.'''

        pass

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

class SklearnTopicModels():
    '''Class that wraps an LDA model in a pipeline.'''

    def __init__(self, n_components=10, estimator='LDA'):
        self.n_components = n_components
        if estimator.upper() == 'LDA':
            self.estimator = LatentDirichletAllocation(n_components=self.n_components)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = TruncatedSVD(n_components=self.n_components)
        self.model = Pipeline([
            ('norm', TextNormalizer('consumer_complaint_narrative')),
            ('vect', TextCountVectorizer(text_col='consumer_complaint_narrative' #tokenizer=identity,
            )),
            ('model', self.estimator)
        ])

    def fit_transform(self, X):
        self.model.fit_transform(X)

        return self.model

    def get_topics(self, n=25):
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.feature_names
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features =topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

def print_topics(dict_like):
    for topic, terms in dict_like.items():
        print(f'Topic {topic+1}')
        print(terms)

base_format = FormatEstimator()
df = base_format.df

# base_format.fit(df)
df = base_format.fit_transform(df)
df_sub = df.loc[:100].copy()

# LDA model:
# lda = SklearnTopicModels(n_components=10, estimator='lsa')
# lda.fit_transform(df_sub)
# lda.model['vect'].feature_names
# topics = lda.get_topics()
# print_topics(topics)




pipe = Pipeline(steps=[
    ('feat_trans', FeatureEngTransformer())
    ,('normalize', TextNormalizer('consumer_complaint_narrative'))
    ,('one_hot', OneHotVectorizer('consumer_complaint_narrative'))
    # ,('kmeans', KMeansEstimator(k=3))# Not this kmeans estimator, for some reason it doesn't work correctly.
    ,('kmeans',KMeansClusters(10) )
])

pipe.fit(df_sub)
clusters = pipe.transform(df_sub)
df_sub['predictions'] = clusters


hierarchy_pipe = Pipeline(steps=[
    ('feat_trans', FeatureEngTransformer())
    ,('normalize', TextNormalizer('consumer_complaint_narrative'))
    ,('one_hot', OneHotVectorizer('consumer_complaint_narrative'))
    ,('clusters', HierarchicalClusters() )
])

hierarchy_pipe.fit(df_sub)
clusters = hierarchy_pipe.transform(df_sub)
hierarchy_pipe['clusters'].children
children = hierarchy_pipe.named_steps['clusters'].children
children[:][0]
plot_dendrogram(children)
# labels = hierarchy_pipe['clusters'].y # This is the the same as the clusters above. Therefore its not needed

df_sub['predictions_hierarchy'] = clusters
df_sub[df_sub.predictions_hierarchy==0].issue.value_counts()
df_sub[df_sub.predictions_hierarchy==1].issue.value_counts()

# From textbook ch.7; Context Free Grammar:
GRAMMAR = """
 S -> NNP VP
 VP -> V PP
 PP -> P NP
 NP -> DT N
 NNP -> 'Gwen' | 'George'
 V -> 'looks' | 'burns'
 P -> 'in' | 'for'
 DT -> 'the'
 N -> 'castle' | 'ocean'
 """
cfg = CFG.fromstring(GRAMMAR)
print(cfg)

from nltk.chunk.regexp import RegexpParser
GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = RegexpParser(GRAMMAR)


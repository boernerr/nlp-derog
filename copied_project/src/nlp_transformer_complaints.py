import string
import re
import os
import pandas as pd
import unicodedata

import nltk
from nltk import pos_tag, wordpunct_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from copied_project.src import DATA_PATH
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

    # def _normalize_old(self, document):
    #     """Normalize the text by lemmatization by removing stopwords and punct."""
    #     document = pos_tag(wordpunct_tokenize(document))
    #     return [self.lemmatize(token, tag).lower()
    #             for sentence in document
    #             for (token,tag) in sentence
    #             if not self.is_punct(token) and not self.is_stopword(token)
    #     ]

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
        """Create a new column in X that is normalized version of self.text_col."""

        X[self.text_col] = X[self.text_col].apply(lambda x: self.normalize(x))
        X[self.text_col] = X[self.text_col].apply(lambda x: [' '.join(i for i in x)])
        return X
        # for doc in X:
        #     yield self.normalize(doc)

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
    '''Class for one-hotting words(i.e. features) into an array.'''

    def __init__(self, text_col):
        self.vectorizer = CountVectorizer(input='content', decode_error='ignore', binary=True)
        self.text_col = text_col

    def fit(self, X, y=None):
        '''Generic fit function.'''

        return self

    def transform(self, X):
        '''Vectorize self.text_col into an array of arrays. '''

        # This works, BUT only vectorizes individual rows. Meaning, it doesn't make one big array vectorizing the entire corpus
        # X['freq_vector'] = X[self.text_col].apply(lambda x: self.vectorizer.fit_transform(x))
        # return X
        X[self.text_col]
        freqs = self.vectorizer.fit_transform(X[self.text_col])
        return freqs




base_format = FormatEstimator()
df = base_format.df

base_format.fit(df)
df = base_format.transform(df)
df_sub = df.loc[:100].copy()

pipe = Pipeline(steps=[
    ('feat_trans', FeatureEngTransformer())
    ,('normalize', TextNormalizer('consumer_complaint_narrative'))
])

pipe.fit(df_sub)
df_sub_xprime = pipe.transform(df_sub)

one_hot = OneHotVectorizer('consumer_complaint_narrative_norm')
one_hot.fit(df_sub_xprime)
one_hot.transform(df_sub_xprime)
test = one_hot.vectorizer.fit_transform(df_sub_xprime['consumer_complaint_narrative_norm'])





one_hot.fit_transform(df_sub_xprime)

one_hot.vectorizer.get_feature_names()
df_sub_xprime['freq_vector'].loc[0].toarray()[0]



normalizer = text_normalizer.TextNormalizer('consumer_complaint_narrative')
normalizer.fit(xprime)
normalizer.transform(xprime)
xprime


# Assume first_person MEMINCIDENTREPORT is about the author!
first_person = xprime[xprime.pcnt_first_pron > .5]
# first_person[first_person.CURTICKETAMOUNT >0].shape[0]/first_person.shape[0] # 2670 total, = 22% of all first_person
# Of the first person records that have a ticket amount, I want to see if any are actually regarding a relative or friend and NOT the author.
first_person_tkt_rela = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_immediate_rela > 0)] # idx regarding rela [6907, 21693, 37610]; 3/168 records are NOT about the author. Stands to reason these are ALL about the author
first_person_tkt_frnd = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_frnd > 0 )] # idx regarding frnd [188, 21693]
first_person_tkt_dstnt = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_dstnt_rela > 0 )] # all idx regarding author


# Assume third_person MEMINCIDENTREPORT is about someone OTHER than the author!
third_person = xprime[xprime.pcnt_first_pron < .25]

third_person_tkt_rela = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_immediate_rela > 0)] # idx regarding rela []
third_person_tkt_frnd = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_frnd > 0 )] # idx regarding frnd [1801, 6431, 8056]
third_person_tkt_dstnt = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_dstnt_rela > 0 )] # only 2/4 are regarding author


# Assume pronoun_na is about the author! After analysis, this holds up
pronoun_na = xprime[xprime.pcnt_first_pron==0]
pronoun_na[pronoun_na.CURTICKETAMOUNT >0].shape[0]/pronoun_na.shape[0] # 46%

pron_na_wRelative = pronoun_na[(pronoun_na.cnt_immediate_rela > 0) | (pronoun_na.cnt_dstnt_rela > 0) | (pronoun_na.cnt_frnd > 0 )]
pron_na_no_Relative = pronoun_na[(pronoun_na.cnt_immediate_rela == 0) & (pronoun_na.cnt_dstnt_rela == 0) & (pronoun_na.cnt_frnd == 0)]


relative = xprime[(xprime.cnt_immediate_rela >0) | (xprime.cnt_dstnt_rela >0)]
relative_high_count = relative[relative.cnt_immediate_rela>7]

################################################################################################
sample1 = pronoun_na.sample(frac=.1, random_state=0) ############################################ This is where I left off
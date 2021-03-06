import string
import re
import os
import pandas as pd

import nltk
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from copied_project.src import DATA_PATH

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

    def __init__(self,
                 path_ = os.path.join(DATA_PATH, 'raw', 'SRSP_SELF_REPORT__lawEnforceInteraction.csv'),
                 ):
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
        '''Create columns for count of occurrences of 1st person pronoun and 3rd person pronouns.
        We would assume a MEMINCIDENTREPORT with a ratio of more 1st PRON than 3rd PRON would heavily indicate the MEMINCIDENTREPORT
        is a description of the author.
        That leaves us '''

        x_prime = X.copy()
        # x_prime['normalized'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._normalize_text(x_))

        x_prime['cnt_third_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._is_third_pers_PRN(x_))
        x_prime['cnt_first_pron'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._is_first_pers_PRN(x_))
        x_prime['cnt_immediate_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._is_relative_immediate(x_))
        x_prime['cnt_dstnt_rela'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._is_relative_dist(x_))
        x_prime['cnt_frnd'] = x_prime.MEMINCIDENTREPORT.apply(lambda x_: self._is_friend(x_))
        x_prime['has_fine'] = x_prime.CURTICKETAMOUNT.apply(lambda x_: 1 if not x_ else 0)


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


base_format = FormatEstimator()
df = base_format.df
base_format.fit(df)
df = base_format.transform(df)


featTrans = FeatureEngTransformer()
featTrans.fit(df)

xprime = featTrans.transform(df)

# Assume first_person MEMINCIDENTREPORT is about the author!
first_person = xprime[xprime.pcnt_first_pron > .5]
# first_person[first_person.CURTICKETAMOUNT >0].shape[0]/first_person.shape[0] # 2670 total, = 22% of all first_person
# Of the first person records that have a ticket amount, I want to see if any are actually regarding a relative or friend and NOT the author.
first_person_tkt_rela = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_immediate_rela > 0)] # idx regarding rela [6907, 21693, 37610]; 3/168 records are NOT about the author. Stands to reason these are ALL about the author
first_person_tkt_frnd = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_frnd > 0 )] # idx regarding frnd [188, 21693]
first_person_tkt_dstnt = first_person[(first_person.CURTICKETAMOUNT >0) & (first_person.cnt_dstnt_rela > 0 )] # all idx regarding author


# Assume third_person MEMINCIDENTREPORT is about someone OTHER than the author!
third_person = xprime[xprime.pcnt_first_pron < .5]

third_person_tkt_rela = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_immediate_rela > 0)] # idx regarding rela []
third_person_tkt_frnd = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_frnd > 0 )] # idx regarding frnd [1801, 6431, 8056]
third_person_tkt_dstnt = third_person[(third_person.CURTICKETAMOUNT >0) & (third_person.cnt_dstnt_rela > 0 )] # only 2/4 are regarding author


# Assume pronoun_na is about the author! After analysis, this holds up
pronoun_na = xprime[xprime.pcnt_first_pron==0]
pronoun_na[pronoun_na.CURTICKETAMOUNT >0].shape[0]/pronoun_na.shape[0] # 46%

pron_na_wRelative = pronoun_na[(pronoun_na.cnt_immediate_rela > 0) | (pronoun_na.cnt_dstnt_rela > 0) | (pronoun_na.cnt_frnd > 0 )]
pron_na_no_Relative = pronoun_na[(pronoun_na.cnt_immediate_rela == 0) & (pronoun_na.cnt_dstnt_rela == 0) & (pronoun_na.cnt_frnd == 0)]

################################################################################################
sample1 = pronoun_na.sample(frac=.1, random_state=0) ############################################ This is where I left off
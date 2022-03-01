import pandas as pd
import nltk
import re
import os
import time
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
# from importlib import reload
# import gensim

# from copied_project.src import _util_plot_funcs as utilf
from copied_project.src import DATA_PATH
from copied_project.src.nlp_transformer_complaints import TextNormalizer, FormatEstimator
# from copied_project.src import nlp_transformer_complaints
# reload(nlp_transformer_complaints)

class TextProcessor(CorpusReader,CategorizedCorpusReader): # holding these inherited classes for now, not sure if they actually are needed

    def __init__(self, df, txt_col):
        self.df = df
        self.txt_col = txt_col
        self._run_preprocess()
        self.counts = nltk.FreqDist()
        self.tokens = nltk.FreqDist()
        self.raw_corpus = [i for i in self.df[self.txt_col].apply(lambda x: self.corpus_tokenize(x))] # format: list of lists
        self.stem_corpus = [' '.join(phrase) for phrase in self.raw_corpus] # stemmed version of raw_corpus. Ready to be vectorized

    def sents(self):
        """Parse text into sentences."""
        self.df['sentences'] = self.df[self.txt_col].apply(lambda x: [sent for sent in sent_tokenize(x)])

    def words(self):
        self.df['tokens'] = self.df[self.txt_col].apply(lambda x:  wordpunct_tokenize(x))

    def tokenize(self):
        """Assign PartOfSpeech to tokenized words."""
        start = time.time()
        self.df['pos'] = self.df[self.txt_col].apply(lambda x: pos_tag(wordpunct_tokenize(x)))
        print(f'tokenize() run in {round(time.time()- start, 2)} seconds')

    def corpus_tokenize(self, text):
        """Tokenizes and stems a given text input."""
        # stem = nltk.stem.SnowballStemmer('english')
        # text = text.lower()
        # for token in nltk.word_tokenize(text):
        #     if token in string.punctuation: continue
        #     # yield stem.stem(token) # The book uses yield
        #     return stem.stem(token)
        stem = nltk.stem.SnowballStemmer('english')
        text_lower = text.lower()
        return [stem.stem(token) for token in nltk.word_tokenize(text_lower)]

    def _run_preprocess(self):
        self.sents()
        self.words()
        self.tokenize()

    def describe(self):
        """Descriptive metrics for corpus."""
        started = time.time()

        # Don't have paragraphs in this dataset- just sentences and POS tagging. (the book has a counter for paragraphs)
        self.counts['sents'] = self.df.sentences.apply(lambda x: len(x)).sum()
        self.counts['words'] = self.df.tokens.apply(lambda x: len(x)).sum()

        for i in self.df.pos:
            for word,tag in i:
                self.tokens[word] +=1
        return {
            'sents': self.counts['sents'],
            'words': self.counts['words'],
            'vocab': len(self.tokens),
            'lex_div': round(float(self.counts['words']) / float(len(self.tokens)), 4),
            'secs': round(time.time() - started, 3)
        }

    def process(self):
        """From the book, this tokenizes each fileid and dumps to a pickle file. Shouldn't be needed for our work."""
        target = os.path.join(DATA_PATH, 'processed')

    def transform(self):
        """From the book, iterates through fileids and calls process() on each fileid. Shouldn't be needed"""
        pass

    def vectorize(self, df):
        features = defaultdict(int)
        for token in self.df[self.txt_col].apply(lambda x: self.corpus_tokenize(x)):
            features[token] +=1
        return features

base_format = FormatEstimator()
df = base_format.df
df = base_format.fit_transform(df)
df_sub = df.iloc[:100].copy()
df_copy = df_sub.copy()

# df.columns: ['complaint_id', 'date_received', 'issue', 'sub_issue','consumer_complaint_narrative', 'state']

df_phone = df[df.consumer_complaint_narrative.str.contains('phone')]
# df_phone.issue.value_counts().plot(kind='barh')
df.issue.value_counts()

import tensorflow

add_to_path = r'C:\Users\Robert\AppData\Roaming\Python\Python37\Scripts'

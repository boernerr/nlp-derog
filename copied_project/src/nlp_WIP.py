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
from importlib import reload
# import gensim

from copied_project.src import _util_plot_funcs as utilf
from copied_project.src import DATA_PATH
from copied_project.src import text_normalizer
reload(text_normalizer)

file = r'C:\Users\Robert\Downloads\complaints-2021-08-30_12_09.csv'
# complaints_entire = pd.read_csv(file)
# complaints_entire = format_df(complaints_entire)
# complaints_entire.to_csv(os.path.join(DATA_PATH,'raw',os.path.basename(file)), index=False)
complaints = pd.read_csv(file)
# complaints.Issue.value_counts()

def convert_to_underscore(stringList):
    newList = []
    for string in stringList:
        # Convert spaces(' ') and dashes('-') to '_'
        newList.append(re.sub(' |-','_',string).lower())
    return newList

def format_df(df):
    new_col = convert_to_underscore(df.columns)
    df.rename(columns=dict(zip(df.columns, new_col)), inplace=True)
    return df

complaints = format_df(complaints)
complaints = complaints.loc[complaints.consumer_complaint_narrative.notna()][['date_received','issue','consumer_complaint_narrative']]
complaints_subset = complaints.copy().loc[:100]


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


text = TextProcessor(complaints_subset, txt_col='consumer_complaint_narrative')
df = text.df
df.issue.value_counts()
text.raw_corpus[0]
# def distinct_tags():
#     tags = []
#     for item in df.pos[0]:
#         if item[1] not in tags:
#             tags.append(item[1] )
#     return tags
# tags = distinct_tags()
textnorm = text_normalizer.TextNormalizer()
df['lemmatized'] = df.consumer_complaint_narrative.apply(lambda x: textnorm.normalize(x))
df['lemmatized'][96]

vectorizer = CountVectorizer()# can use following argument for OneHotting: binary=True
X = vectorizer.fit_transform(text.stem_corpus)
print(vectorizer.get_feature_names())
len(vectorizer.get_feature_names())

tfidf = TfidfVectorizer()
corpus = tfidf.fit_transform(text.stem_corpus)
tfidf.get_feature_names()
corpus.shape
corpus[1]


normalizer = text_normalizer.TextNormalizer()

normalizer.lemmatize(df.pos[0][-2][0],df.pos[0][-2][1])
lemma = nltk.WordNetLemmatizer()
lemma.lemmatize('agreement',wn.NOUN)

corpus = [
 "The elephant sneezed at the sight of potatoes.",
 "Bats can see via echolocation. See the bat sight sneeze!",
 "Wondering, she opened the door to the studio.",
]
sample_sent = '''I have a prepaid credit card with capital one. I always pay my bill on time, and my payment posts to my account and the money is taken from my bank.'''
tokenized = text.corpus_tokenize(sample_sent)
vectors = vectorizer.fit_transform(corpus)
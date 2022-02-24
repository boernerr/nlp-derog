import pandas as pd
import numpy as np
from tqdm import tqdm

from keras.models import Input, Model
from keras.layers import Dense

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from scipy import sparse
from sklearn.pipeline import Pipeline
# project imports
from copied_project.src import DATA_PATH, MODULE_PATH
from copied_project.src.nlp_transformer_complaints import ( TextNormalizer) #FormatEstimatorRussia  , CondenseTransformer
from copied_project.src.nlp_transformer import StopWordExtractorTransformer
# Create custom word embeddings based on our SRSP/SIRS data.

class Word2VecNormalizer(TextNormalizer):
    ''''''

    def __int__(self):
        super().__init__()

    def sent_tokens(self, X):
        return sent_tokenize(X)

    def normalize(self, document):
        """
        Normalize the text by lemmatization by removing stopwords and punct while maintaining sentence structure.
        This function should be implemented by:
        pd.Series.apply(lambda x: normalize(x))

        Param
        -----
        document : pd.Series
            In this case, we are applying normalize() to a pd.Series in a pd.DataFrame. Each row, then, is a distinct 'document'.
            The pd.Series in question should be one long string.

        """
        # append each sentence to this list so that we can keep distinct sentences
        sents = []
        for sentence in document:
            doc = pos_tag(wordpunct_tokenize(sentence))
            # print([k for k in doc])
            # j[0] = token, j[1] = tag
            sents.append( [self.lemmatize(j[0].lower(), j[1]) for j in [k for k in doc]
                    if not self.is_punct(j[0]) and not self.is_stopword(j[0])])
        return sents

    def transform(self, X):
        X[self.text_col] = X[self.text_col].apply(lambda token: self.sent_tokens(token))
        X[self.text_col] = X[self.text_col].apply(lambda x: self.normalize(x))
        # This outputs text to be one long str
        if self.return_type == 'str':
            X[self.text_col] = X[self.text_col].apply(lambda x: ' '.join(i for i in x))
        return X

pipe_format = Pipeline([('format_estimator', FormatEstimatorRussia()),
                         ('condense', CondenseTransformer())])
# We first normalize the text by lemmatizing words and removing punctuation and stopwords:
pipe_normalize = Pipeline([('normalize', Word2VecNormalizer(text_col='consumer_complaint_narrative'))
                 # ('stop_words', StopWordExtractorTransformer(text_col='ALLEGATION'))
                          ])

#Create our data:
formatter = FormatEstimatorRussia()
df_russ = formatter.df

df = pipe_format.fit_transform(df_russ)
df_copy = df.iloc[:2500].copy()

df_sub = df.iloc[:2].copy()


class WordMatrix():
    '''Preprocess data to be ingested into a neural network.'''

    def __init__(self, window=2, text_col='consumer_complaint_narrative'):
        self.text_col = text_col
        self.window = window
        self.word_lists = []
        self.all_text = []
        self.uniq_words = []
        self.uniq_wrd_dict = dict()
        self.n_words = 0 #len(self.uniq_words)
        self.X_matrix = []
        self.Y_matrix = []

    def word_neighbors(self, text):
        '''Appends a target word and its context words to a master list of words.'''
        for sentence in text:
            for i, word in enumerate(sentence):
                for w in range(self.window):
                    if i + 1 + w < len(sentence):
                        # Get the words ahead of the target word within the window
                        self.word_lists.append([word] + [sentence[(i + 1 +w)]])
                    if i - w - 1 >=0:
                        self.word_lists.append([word] + [sentence[(i - w - 1)]])

    def _add_uniq_words(self, text):
        '''Dictionary containing all unique words within the text column.'''
        for sentence in text:
            for word in sentence:
                if word not in self.uniq_words:
                    self.uniq_words.append(word)

    def _make_uniq_word_dict(self):
        '''Convert self.uniq_words into a dict().'''
        self.n_words = len(self.uniq_words)
        self.uniq_words.sort()
        # self.uniq_wrd_dict = dict(sorted(zip(self.uniq_words, range(len_words)), key= lambda x: x[0])) # Not needed, below line works
        self.uniq_wrd_dict = dict(zip(self.uniq_words, range(self.n_words)))

    def create_matrices(self):
        for i, word_pair in enumerate(self.word_lists):
            main_word_index = self.uniq_wrd_dict.get(word_pair[0])
            context_word_index = self.uniq_wrd_dict.get(word_pair[1])
            # Placeholders for matrix size
            X_row = np.zeros(self.n_words)
            Y_row = np.zeros(self.n_words)

            # One hot encoding @ position of main word:
            X_row[main_word_index] = 1
            # One hot encoding @ position of context word:
            Y_row[context_word_index] = 1
            # Appending to main matrices:
            self.X_matrix.append(X_row)
            self.Y_matrix.append(Y_row)
        # Convert to sparse matrices
        self.X_matrix = sparse.csr_matrix(self.X_matrix)
        self.Y_matrix = sparse.csr_matrix(self.Y_matrix)

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        '''Create pairs of center word and context word within window parameter.'''

        # Create the (context word, center word) pairs
        df[self.text_col].apply(lambda x: self.word_neighbors(x))
        # Create unique word list
        df[self.text_col].apply(lambda x: self._add_uniq_words(x))
        # Create unique word dictionary
        self._make_uniq_word_dict()


# Return text column as list of tokens with stop words removed:
pipe_normalize.fit_transform(df_copy)

word = WordMatrix()
word.transform(df_copy)

word.uniq_wrd_dict
len(word.word_lists)
word.n_words

word.create_matrices()

embed_size = 2

# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')


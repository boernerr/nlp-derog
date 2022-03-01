import ast
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import sys

from collections import Counter
from datetime import datetime
from os import path
from tqdm import tqdm
from time import perf_counter
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from scipy import sparse
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.pipeline import Pipeline
# project imports
from nlp_derog import DATA_PATH, MODULE_PATH
from nlp_derog.src.transformers import ( TextNormalizer, FormatEstimatorRussia, CondenseTransformer)


class Word2VecNormalizer(TextNormalizer):
    '''Normalize text by lemmatization and stopword removal.

    This class inherits from the TextNormalizer class and add functionality for returning each record within
    self.text_col as a list of sentences (if there are > 0 sentences). The purpose of this class is to preserve the
    sentence structure of self.text_col by indicating the boundary between sentences. The returned list of sentences can
    then be used downstream for tasks such as word co-occurrences, which otherwise would be inaccurate if a simple bag-of-words
    (BOW) methodology was used.
    '''

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

class WordMatrix():
    '''Preprocess data to be ingested into a neural network.

    This class ingests text as a list of sentences, and outputs sparse matrices based upon the co-occurrence of words
    within the specified window parameter. I.e. if window=2, for each given word (the 'center word') we will denote a
    co-occurrence with other words that occur positionally within [word -2 and word +2].

    Example sentence:
        'George veraciously confirmed the fact that Al Gore is undeniably man-bear-pig.'

        We will assume that our center word from the above = 'fact' and the window parameter = 2. After stopword removal,
        the returned list of co-occurrences will be as follows:
        [
            ['fact', 'confirmed'],
            ['fact', 'veraciously'],
            ['that','fact'],
            ['Al','fact']
        ]
    '''
    test_var = 0

    def __init__(self, window=2, text_col='ALLEGATION'):
        self.text_col = text_col
        self.window = window
        self.word_lists = [] # list of bigrams for a center word and all words within its window
        self.all_sentences = []
        self.all_sentences_encoded = []
        self.uniq_words = []
        self.uniq_wrd_dict = dict()
        self.idx2word = dict()
        self.bag_of_words = [] # Don't really need this as an instance variable
        self.n_words = 0 #len(self.uniq_words), count of uniq words
        self.word_counts = None # The frequency for all words within the entire corpus
        self.X_matrix = []
        self.Y_matrix = []
        WordMatrix.test_var += 1 # Test if this increments self.test_var by 1 on each instantiation of this class

    def _master_sentence_list(self, text):
        '''Create master list of all sentences in data set/corpus.'''
        for sentence in text:
            self.all_sentences.append(sentence)

    def _encoded_master_sentence_list(self, text):
        '''Create master list of all sentences encoded to an integer in data set/corpus.'''
        for sentence in text:
            sent = [self.uniq_wrd_dict[w] for w in sentence]
            self.all_sentences_encoded.append(sent)

    def _count_uniq_words(self, text):
        '''Retrieve the occurrence of all words in the text_col for the entire data set/corpus.'''
        for sentence in text:
            for word in sentence:
                self.bag_of_words.append(word)
        # Sort words in descending order:
        self.word_counts = dict(Counter(self.bag_of_words).most_common())

    def word_neighbors(self, text):
        '''THIS NEEDS TO BE MODIFIED!!

        Appends a target word and its context words to a master list of words.
        '''
        for sentence in text:
            for i, word in enumerate(sentence):
                for w in range(self.window):
                    if i + 1 + w < len(sentence):
                        # Get the words ahead of the target word within the window
                        self.word_lists.append([word] + [sentence[(i + 1 +w)]])
                    if i - w - 1 >=0:
                        self.word_lists.append([word] + [sentence[(i - w - 1)]])

    def _add_uniq_words(self, text):
        '''Dictionary containing all unique words mapped to a unique integer within the text column.'''
        for sentence in text:
            for word in sentence:
                if word not in self.uniq_words:
                    self.uniq_words.append(word)

    def _make_uniq_word_dict(self):
        '''Convert self.uniq_words into a dict() and create the reversed dict().'''
        self.n_words = len(self.uniq_words)
        self.uniq_words.sort()
        # self.uniq_wrd_dict = dict(sorted(zip(self.uniq_words, range(len_words)), key= lambda x: x[0])) # Not needed, below line works
        self.uniq_wrd_dict = dict(zip(self.uniq_words, range(self.n_words)))
        self.idx2word = dict(zip(range(self.n_words), self.uniq_words)) # dict(map(reversed, self.uniq_wrd_dict.items()))

    def create_matrices(self):
        '''This method is from github.com/Eligijus112. This assumes the use of Tensorflow and Keras.

        THIS SHOULD NOT BE USED AT THIS TIME!!!
        '''
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

    def _check_text_type(self, df):
        '''Check that self.text_col is congruous with expected data type.'''
        content = df[self.text_col].iloc[0]
        if not isinstance(content, list):
            try:
                # Print messages only for debugging:
                # print(f'input data type for "df.{self.text_col}" == {type(content)}')
                df[self.text_col] = df[self.text_col].apply(lambda x: ast.literal_eval(x))
                # print(f'output data type for "df.{self.text_col}" == {type(df[self.text_col].iloc[0])}')
            except TypeError:
                print(f'The data type within {self.text_col} should either be str, list, or pd.Series; got {type(self.text_col)} instead.')

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        '''Create pairs of center word and context word within window parameter.'''
        self._check_text_type(df)
        # Create master list of sentences; this may not be needed in favor of _encoded_master_sentence_list()
        df[self.text_col].apply(lambda x: self._master_sentence_list(x))
        # Create counter object to count word freqs
        df[self.text_col].apply(lambda x: self._count_uniq_words(x))
        # Create the (context word, center word) pairs
        df[self.text_col].apply(lambda x: self.word_neighbors(x))
        # Create unique word list
        df[self.text_col].apply(lambda x: self._add_uniq_words(x))
        # Create unique word dictionary
        self._make_uniq_word_dict()
        # Create master list of encoded sentences, this is dependent on self._add_uniq_words() running prior
        df[self.text_col].apply(lambda x: self._encoded_master_sentence_list(x))


pipe_format = Pipeline([('format_estimator', FormatEstimatorRussia()),
                         ('condense', CondenseTransformer())])
# We first normalize the text by lemmatizing words and removing punctuation and stopwords:
pipe_normalize = Pipeline([('normalize', Word2VecNormalizer(text_col='ALLEGATION', return_type=None))
                 # ('stop_words', StopWordExtractorTransformer(text_col='ALLEGATION'))
                          ])

#Create our data:
formatter = FormatEstimatorRussia()
df_russ = formatter.df

df = pipe_format.fit_transform(df_russ)

df_sub = df.iloc[:1000].copy()
df_2row = df.iloc[:20].copy()

# Return text column as list of tokens with stop words removed and output to csv:
pipe_normalize.fit_transform(df_sub)
# today = datetime.today().strftime('%m_%d_%Y')
# df.to_csv(path.join(MODULE_PATH, 'model', f'SRSP_formatted_for_neural_net_{today}.csv'))
# neural_net_df = pd.read_csv(path.join(MODULE_PATH, 'model', f'SRSP_formatted_for_neural_net_02_17_2022.csv'))

word = WordMatrix()
word.transform(df_sub)

# Testing class instance vars
word.uniq_wrd_dict['dad']
word.uniq_wrd_dict
word.idx2word
word.all_sentences
sum(word.word_counts.values())




# Hard coding neural net:
class TrainNeuralNet():

    def __init__(self, matrix_object):
        self.matrix_object = matrix_object
        self.window_size = 5
        self.learning_rate = .025
        self.final_learning_rate = 1e-4
        self.num_negatives = 5
        self.epochs =20
        self.D = 50 # word embedding size, i.e. the dimensionality of the vector for each word. Vector with shape = ( , 50)
        self.learning_rate_delta = (self.learning_rate - self.final_learning_rate)/ self.epochs # learning rate decay
        self.W = np.random.randn(self.matrix_object.n_words, self.D)
        self.V = np.random.randn(self.D, self.matrix_object.n_words)
        self.p_neg = self.get_negative_sampling_distr() #sentences, vocab_size # sentences = each distinct sentence; should be WordMatrix.word_lists object
        self.threshold = 1e-5
        self.p_drop = 1 - np.sqrt(self.threshold / self.p_neg)
        self.costs = []

    def get_negative_sampling_distr(self):
        vocab_size = self.matrix_object.n_words
        sentences = self.matrix_object.all_sentences # This should be OK to use because word_idx var below gets the int value of word
        word_freq = np.zeros(vocab_size)
        word_count = sum(len(sentence) for sentence in sentences) # this is equivalent to sum(self.matrix_object.word_counts.values())

        for sent in sentences:
            for word in sent:
                word_idx = self.matrix_object.uniq_wrd_dict[word] # This expects our sentences to be the acutal sentences with words, not words mapped to int
                word_freq[word_idx] += 1
        # Smooth word_freq, raise to the power of 0.75 because this is current standard
        p_neg = word_freq**0.75

        # normalize p_neg
        p_neg = p_neg / p_neg.sum()
        assert(np.all(p_neg > 0))
        return p_neg

    def get_context(self, pos, sentence, window_size):
        '''Retrieves the context words within a window around a center word.'''
        start = max(0, pos - window_size)
        end_ = min(len(sentence), pos + window_size)

        context = []
        for ctx_pos, ctx_word_idx in enumerate(sentence[start : end_], start=start):
            if ctx_pos != pos:
                # We're not including the 'center' word as a target
                context.append(ctx_word_idx)

        return context

    def sgd(self, input_, targets, label, learning_rate, W, V):
        '''Perform stochastic gradient descent.'''
        activation = W[input_].dot(V[:, targets])
        prob = sigmoid(activation)
        # Gradients
        gV = np.outer(W[input_], prob - label) # D x N
        gW = np.sum((prob - label) * V[:, targets], axis=1) # D

        V[:, targets] -= learning_rate * gV
        W[input_] -= learning_rate * gW
        # Return cost (binary cross entropy)
        cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
        return cost.sum()

    def transform(self, sentences=None):
        if not sentences:
            sentences = self.matrix_object.all_sentences_encoded
        for epoch in range(self.epochs):
            # shuffle sentence order
            np.random.shuffle(sentences)
            cost = 0
            counter = 0
            t0 = perf_counter()
            for sentence in sentences:
                sentence = [w for w in sentence if np.random.random() < (1 - self.p_drop[w])]
                if len(sentence) < 2:
                    continue

                randomly_ordered_positions = np.random.choice(
                    len(sentence),
                    size=len(sentence),
                    replace=False
                )

                for pos in randomly_ordered_positions:
                    # This is our 'center' word
                    word = sentence[pos]
                    # positive context words/negative samples
                    context_words = self.get_context(pos, sentence, self.window_size)
                    neg_word = np.random.choice(self.matrix_object.n_words, p=self.p_neg)
                    targets = np.array(context_words)

                    # one iter of stochastic gradient descent
                    c = self.sgd(word, targets, 1, self.learning_rate, self.W, self.V)
                    cost += c
                    c = self.sgd(neg_word, targets, 0, self.learning_rate, self.W, self.V)
                    cost += c

                counter += 1
                if counter % 100 == 0:
                    sys.stdout.write(f'processed {counter} / {len(sentences)}')
                    sys.stdout.flush()

            t1 = perf_counter() - t0
            print(f'Epoch complete: {epoch}, cost: {cost}, time: {t1}')
            # Save costs
            self.costs.append(cost)
            # Update learning rate
            self.learning_rate -= self.learning_rate_delta

        plt.plot(self.costs) #, title='Training cost per iteration'

    def analogy(self, pos1, neg1, pos2, neg2, word2idx, idx2word, W):
        '''Input words to test analogy i.e. (man - king :: woman - queen)'''
        V, D = W.shape
        # Print analogies for easy use in console OR could send to log file
        print(f'{pos1} - {neg1} = {pos2} - {neg2}')
        for w in (pos1, neg1, pos2, neg2):
            if w not in word2idx:
                print(f'{w} is not in the corpus of word2idx, choose another word!')
                return

        p1 = W[word2idx[pos1]]
        n1 = W[word2idx[neg1]]
        p2 = W[word2idx[pos2]]
        n2 = W[word2idx[neg2]]

        vec = p1 - n1 + n2

        distances  = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
        # Only pick the top 10 most similar results
        idx = distances.argsort()[:10]

        best_idx = -1
        keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]

        for i in idx:
            if i not in keep_out:
                best_idx = i
                break

        # Optional printing; ease of use in the console
        print(f'got: {pos1} - {neg1} = {idx2word[best_idx]} - {neg2}')
        print('closest 10:')
        for i in idx:
            print(idx2word[i], distances[i])

        print()
        print(f'dist to {pos2}: {cos_dist(p2, vec)}')

    def test_model(self, king, man, queen, woman):
        for We in (self.W, (self.W + self.V.T) / 2):
            self.analogy(king, man, queen, woman, self.matrix_object.uniq_wrd_dict, self.matrix_object.idx2word, We)


word.all_sentences
nn = TrainNeuralNet(word)

nn.transform()
nn.test_model('january', 'february', 'june', 'july')
plt.plot(nn.costs)

word.uniq_wrd_dict['july']
word.idx2word[1833]


# freq_vector = nn.get_negative_sampling_distr()
# sum(freq_vector)
# len(nn.p_neg)

# w2x = {'one': None, 'two': None, 'four': None}
# for w in ('one', 'two', 'three'):
#     if w not in w2x:
#         print(w)

# Following for implementing Tensorflow:
# embed_size = 2
#
# X = word.X_matrix
# Y = word.Y_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# # Defining the neural network
# inp = Input(shape=(X.shape[1],))
# x = Dense(units=embed_size, activation='linear')(inp)
# x = Dense(units=Y.shape[1], activation='softmax')(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
#
# # Optimizing the network weights
# model.fit(
#     x=X,
#     y=Y,
#     batch_size=256,
#     epochs=1000
#     )
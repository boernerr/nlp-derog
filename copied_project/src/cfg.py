from itertools import groupby
from nltk import ne_chunk
from nltk.chunk import tree2conlltags
from nltk import CFG
from nltk.chunk.regexp import RegexpParser
from sklearn.base import BaseEstimator,TransformerMixin
from unicodedata import category as unicat

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
print(cfg.start())
print(cfg.productions())

# We need both a grammar and parser in order to define POS patterns AND chunk the patterns.
GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
# chunker = RegexpParser(GRAMMAR)
# book_test = ('''Dusty Baker pro‐
# posed a simple solution to the Washington National’s early-season bullpen troubles
# Monday afternoon and it had nothing to do with his maligned group of relievers.''')
GOODTAGS = frozenset(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])

class KeyphraseExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, text_col, grammar=GRAMMAR ):
        self.grammar = GRAMMAR
        self.chunker = RegexpParser(self.grammar)
        self.text_col = text_col

    def normalize(self, sent):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        is_punct = lambda word: all(unicat(c).startswith('P') for c in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_keyphrases(self, document):
        """
        For a document, parse sentences using our chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Yields extracted phrases.
        """
        for sents in document:
            for sent in sents:
                sent = self.normalize(sent)
            if not sent:
                continue
        chunks = tree2conlltags(self.chunker.parse(sent))
        phrases = [
            " ".join(word for word, pos, chunk in group).lower()
            for key, group in groupby(
                chunks, lambda term: term[-1] != 'O'
            ) if key
        ]
        for phrase in phrases:
            yield phrase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['keyphrases'] = X[self.text_col].apply(lambda x: self.extract_keyphrases(x))
        # yield self.extract_keyphrases(document)

GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])

class EntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, labels=GOODLABELS, **kwargs):
        self.labels = labels

    def get_entities(self, document):
        entities = []
        for paragraph in document:
            for sentence in paragraph:
                trees = ne_chunk(sentence)
                for tree in trees:
                    if hasattr(tree, 'label'):
                        if tree.label() in self.labels:
                            entities.append(
                        ' '.join([child[0].lower() for child in tree])
                        )
        return entities

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.get_entities(document)


# rand_ser = pd.Series([np.random.randint(0,6) for i in range(20)])
# # np.digitize(rand_ser, [0, 3, 4, 5])
# dff = pd.DataFrame(rand_ser, columns=['rand_int'])
# dff['bins'] = dff.rand_int.apply(lambda x: np.digitize(x, [0, 3, 4, 5]))

from keras.layers import Dense
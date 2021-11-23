import unicodedata
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, wordpunct_tokenize
from sklearn.base import BaseEstimator,TransformerMixin

class TextNormalizer(BaseEstimator,TransformerMixin):

    def __init__(self):
        self.language = 'english'
        self.stopwords = set(nltk.corpus.stopwords.words(self.language))
        self.lemmatizer = nltk.WordNetLemmatizer()

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

    def transform(self, documents):
        """Generic transform func() to comply with SKLEARN."""
        for doc in documents:
            yield self.normalize(doc)


# textnorm = TextNormalizer()
# dfcopy = df.copy().loc[:10]
# dfcopy['lemmatized'] = dfcopy.consumer_complaint_narrative.apply(lambda x: textnorm.normalize(x))
# dfcopy['lemmatized'][0]

# smpl_txt = '''I have a prepaid credit card with capital one.
#             I always pay my bill on time, and my payment posts to my account and the money is taken from my bank.'''

# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
#
# model = Pipeline([
#  ('normalizer', TextNormalizer()),
#  # ('vectorizer', GensimVectorizer()),
#  ('bayes', MultinomialNB()),
# ])
#
# model.named_steps['bayes']








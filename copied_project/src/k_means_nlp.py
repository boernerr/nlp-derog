import nltk.cluster.util
from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin

class KMeansClusters(BaseEstimator, TransformerMixin):

    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance()
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """Fits k-means to one-ht encoded vectorize documents."""
        return self.model.cluster(documents, assign_clusters=True)
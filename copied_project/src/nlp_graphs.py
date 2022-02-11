import networkx as nx
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt

def graph_synsets(terms, pos=wn.NOUN, depth=2):
    """
    Create a networkx graph of the given terms to the given depth.
    """
    G = nx.Graph(
        name="WordNet Synsets Graph for {}".format(", ".join(terms)), depth=depth,
    )

    def add_term_links(G, term, current_depth):
        for syn in wn.synsets(term):
            for name in syn.lemma_names():
                G.add_edge(term, name)
                if current_depth < depth:
                    add_term_links(G, name, current_depth + 1)

    for term in terms:
        add_term_links(G, term, 0)
    return G


def draw_text_graph(G):
    pos = nx.spring_layout(G, scale=18)
    nx.draw_networkx_nodes(
        G, pos, node_color="white", linewidths=0, node_size=500
    )
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='lightgrey')
    plt.tick_params(
         axis='both', # changes apply to both the x- and y-axis
         which='both', # both major and minor ticks are affected
         bottom='off', # turn off ticks along bottom edge
         left='off', # turn off ticks along left edge
         labelbottom='off', # turn off labels along bottom edge
         labelleft='off') # turn off labels along left edge
    plt.show()

class GraphExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.G = nx.Graph()

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        for document in documents:
            for first, second in document:
                if (first, second) in self.G.edges():
                    self.G.edges[(first, second)]['weight'] += 1
                else:
                    self.G.add_edge(first, second, weight=1)
        return self.G


G = graph_synsets(['officer'])
nx.info(G)
list(G.nodes)
nx.draw(G, with_labels=True)

draw_text_graph(G)


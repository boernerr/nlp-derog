import matplotlib.pyplot as plt
import numpy as np

def date_plot(self, xmin=None, xmax=None, figsize=(15, 7), **kwargs):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        plt.plot_date(self.x, self.y, ls='-', **kwargs)
        plt.xlim(xmin, xmax)
        plt.title(f'Plot for {self.value}')

def bar_plot(data, figsize=(15, 7), r=90, title=None, **kwargs):
    """Accept data as a dict()."""
    X = list(data.keys())
    y = list(data.values())
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        plt.barh(X, y,**kwargs)
        plt.xticks(rotation=r)
        plt.title(f'Plot for {title}')


from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(children, **kwargs):
# Distances between each pair of children
    distance = position = np.arange(children.shape[0])
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([
                        children, distance, position]
                        ).astype(float)
    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(figsize=(10, 5)) # set size
    ax = dendrogram(linkage_matrix, **kwargs)
    plt.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt

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

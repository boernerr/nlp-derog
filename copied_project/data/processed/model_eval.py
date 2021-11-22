
from ds_project_template import DATA_PATH,PROJ_PATH


import pandas as pd
import numpy as np
import seaborn as sns
# from dbaccess import DataReader
import matplotlib.pyplot as plt
from ds_project_template.src.preprocess_util import df_date_mkr
#fPath = r'./model/decoded_model_output.csv'

#below is accessing the data files directory without calling any functions. THis is for EDA only.
from pandas._libs.tslibs.strptime import datetime_date

raw = r'C:\Users\RHBOERNER\Documents\SQL_Extracts/TRAVEL_MASTER2000_2021_extract.csv'
evalDf = pd.DataFrame(pd.read_csv(raw))
evalDf = evalDf[['SSN_MASTER',  'COUNTRY', 'FROM_MONTH', 'FROM_YEAR', 'DISCREPANCY_RECORD']]
df = evalDf.copy()


df = df_date_mkr(df,min_dt='01-01-2011',max_dt='03-01-2020')
df.groupby(df.date.dt.year)['date'].count()

class SeriesPlot():

    def __init__(self, df, col:str, value=None):
        self.df = df
        self.col = col
        self.value = value
        self.x,self.y = self._date_data()

    def _date_data(self):
        if isinstance(self.value, str):
            x, y = np.unique(self.df[self.df[self.col] == f'{self.value}'].date, return_counts=True)
        else:
            x, y = np.unique(self.df[self.df[self.col] == self.value].date, return_counts=True)
        return x,y

    def date_plot(self, xmin=None, xmax=None, figsize = (15,7), **kwargs):
        with plt.style.context('ggplot'):
            fig,ax = plt.subplots(figsize=figsize)
            plt.plot_date(self.x,self.y, ls = '-', **kwargs)
            plt.xlim(xmin,xmax)
            plt.title(f'Plot for {self.value}')


    def hist_plot(self, xlbl:str=None,ylbl:str=None):
        col_vals = np.array(self.df[self.col].value_counts().values)
        maxval= max(col_vals)
        binWidth = np.std(col_vals)/2
        # uniqvals = len(np.unique(col_vals))
        numBins = int(round(maxval/binWidth))

        with plt.style.context('ggplot'):
            fig,ax = plt.subplots(figsize= (15,7))
            sns.distplot(col_vals,bins= numBins, color='navy', kde_kws={'shade':True, 'bw':1}, hist=True)
            plt.xlabel(xlbl,fontsize=15), plt.ylabel(ylbl,fontsize=15)
        #    Below line is BAD use of setting yticks. The param is set for ONLY this specific case and needs to be updated for subsequent use.
            plt.xticks(ticks= np.linspace(0,maxval,numBins),fontsize=10,rotation=90),plt.yticks()#(ticks = np.linspace(0,.3,7),labels = (np.round(np.linspace(0,.3,7)*100)).astype(int),fontsize=15)
            plt.xlim(0,max(col_vals)+1)


SeriesPlot(df,col='SSN_MASTER').hist_plot()

SeriesPlot(df,col='SSN_MASTER',value= 228394626).date_plot()

###################____TESTING____################
def hist_plot(df, col):
    col_vals = np.array(df[col].value_counts().values)
    maxval= max(col_vals)
    bin_siz = int(round(maxval / 4))
    print(col_vals,maxval,bin_siz)

hist_plot(df, 'COUNTRY')
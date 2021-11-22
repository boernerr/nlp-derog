# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:56:30 2021

@author: rhboerner
"""
import datetime
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder


def df_date_mkr(df,min_dt='12-01-2016',max_dt='01-01-2021'):
    '''Filter df to applicable dates'''
    
    df['date'] = df.FROM_MONTH.astype(int).astype(str) +'-01-' + df.FROM_YEAR.astype(int).astype(str)
    df.date = df.date.apply(lambda date: datetime.datetime.strptime(date,'%m-%d-%Y' ) )
    df = df.loc[(df.date > pd.Timestamp(min_dt)) & (df.date < pd.Timestamp(max_dt))]
    df.reset_index(inplace=True)
    return df


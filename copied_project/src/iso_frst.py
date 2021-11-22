from os import path
import numpy as np
import pandas as pd
import pickle
from copied_project.src import DATA_PATH
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

#Estimating % of population that is anomalous. -.0055 value = percent of all polygraphs that resulted in DI.
class IsoForest():

    def __init__(self, df):
        self.df = df
        self.rslt_df = self.df.copy()
        self.X = self.df.to_numpy()
        self.predictions = None
        self._impute_na_columns(self.df)

    def _impute_na_columns(self, df):
        impute_med = SimpleImputer(missing_values=np.nan, strategy='median')
        impute_med.fit(df)
        transformed = impute_med.transform(df)
        imputed_df = pd.DataFrame(transformed, columns=df.columns,
                                   index=df.index)
        self.df = imputed_df

    def model_iterator(self, est_list:list=[100], contam=.0055):
        '''
        Fits Isolation forest to the dataset.

        Param
        -----
        X : np.array
            array containing the data to run model on.
        est_list : list
            List of estimators used for the IsolationForest algorithm.
        contam : float
            Float of estimated percentage of anomalous values present in the data.

        '''
        # self.X = self.df.to_numpy()
        for n in est_list:
            model = IsolationForest(contamination=contam, n_estimators=n)
    #        df = df[['SSN_MASTER', 'COUNTRY', 'DISCREPANCY_RECORD', 'date', 'FROM_MONTH']]
            model.fit(self.df)
            self.predictions = model.predict(self.df)
            self.rslt_df[f'predicted_ifrst_{n}'] = pd.Series(model.predict(self.df),index=self.rslt_df.index)
            # prediction = model.predict(X)
            #mapping output values. default values: anomaly = -1, normal = 1
            self.rslt_df[f'predicted_ifrst_{n}'] = self.rslt_df[f'predicted_ifrst_{n}'].map({1:0,-1:1})
            print(f'\n Isolation Forest model run for: {n} estimators')

    def dict_reader(f:str):
        '''Read dictionary back in from .PICKLE file.'''
        with open(f,'rb') as file:
            loaded = pickle.load(file)
        return loaded

    def run_model():
        try:
            data_path = path.join(DATA_PATH,'processed','preprocessed_data_for_model.csv')
            model_data = pd.read_csv(data_path,index_col=None)
            model_iterator(model_data, est_list=[100])
            model_data.to_csv(path.join(DATA_PATH,'..','model','direct_model_output.csv'),index=False)
            decode = dict_reader(path.join(DATA_PATH,'processed','encoded_dict.pickle'))
            model_data.COUNTRY = model_data.COUNTRY.map(decode['COUNTRY'])
            model_data.date = model_data.date.map(decode['datedict']['decode'])
            model_data.to_csv(path.join(DATA_PATH,'..','model','decoded_model_output.csv'),index=False)
        except(PermissionError) as e:
            print(e,'\n The file you are creating is most likely open')
    #    finally:
    #        print('model run!')


all_labor = all_have_label[all_have_label.category=='LABOR']
all_labor = all_labor[['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj']]
all_labor_not_labor = all_labor[all_labor.predicted!='LABOR']
iforest = IsoForest(all_labor)
iforest.df[iforest.df.AGE.isna()]
# iforest.df.AGE.value_counts(dropna=False)
# all_labor[all_labor.AGE.isna()]

iforest.model_iterator(contam=.05)
iforest.rslt_df

labor = all_have_label.loc[all_have_label.category=='labor'.upper()].copy()
labor = labor.merge(pd.DataFrame(iforest.rslt_df.predicted_ifrst_100, index=iforest.rslt_df.index),
                      how='left', left_index=True, right_index=True)
labor_anoms = labor[labor.predicted_ifrst_100==1]
labor_anoms[labor_anoms.predicted!='LABOR'].position_cleaned
# Labor

labor_anoms.position_cleaned.value_counts()
labor[labor.position_cleaned=='MECHANICAL ENGINEER'].position_cleaned.value_counts() # 76/94
labor[labor.position_cleaned=='AUTOMOTIVE FLEET PROGRAM ANALYST'].position_cleaned.count() # 63/91
labor[labor.position_cleaned=='AIRCRAFT MECHANIC INSPECTOR'].position_cleaned.count() # 30/39
labor[labor.position_cleaned=='MECHANICAL ENGINEER HQ'].position_cleaned.value_counts()# 30/31
labor[labor.position_cleaned=='CUSTODIAL WORKER'].position_cleaned.value_counts() #20/330
labor[labor.position_cleaned=='HUMAN RESOURCES SPECIALIST PERFORMANCE MGMT'].position_cleaned.value_counts() # 7/26
labor[labor.position_cleaned=='MAINTENANCE MECHANIC SUPVR'].position_cleaned.value_counts() # 1/12
labor[labor.position_cleaned=='SMALL ARMS REPAIRER'].position_cleaned.value_counts() # 10/27
labor[labor.position_cleaned=='PHOTOGRAPHER LABORATORY'].position_cleaned.value_counts() # 2/66

# Admin
Admin = all_have_label[all_have_label.category=='administration'.upper()].copy()
all_admin_not_admin = Admin[Admin.predicted!='administration'.upper()]
all_Admin = Admin[['OCC_SERIES', 'COST_CODE', 'salaryCpiAdj']]
frst_Admin = IsoForest(all_Admin)

frst_Admin.model_iterator(contam=.05)
Admin = Admin.merge(pd.DataFrame(frst_Admin.rslt_df.predicted_ifrst_100, index=frst_Admin.rslt_df.index),
                      how='left', left_index=True, right_index=True)
Admin.position_cleaned.value_counts()
Admin_anoms = Admin[Admin.predicted_ifrst_100==1]
Admin_anoms_not_admin = Admin_anoms[Admin_anoms.predicted!='administration'.upper()]
Admin_anoms_not_admin.position_cleaned.value_counts()


it_spec = Admin[Admin.position_cleaned=='SUPERVISORY INFORMATION TECHNOLOGY SPE']
it_spec.predicted.value_counts()
Admin[Admin.position_cleaned=='ADMINISTRATIVE OFFICER'].position_cleaned.value_counts() # 260/658

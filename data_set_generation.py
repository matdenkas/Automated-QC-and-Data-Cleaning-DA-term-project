import numpy as np
import pandas as pd
import json
from urllib.error import HTTPError
from urllib.request import urlopen
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from cust_data_types import Data_Set

MANUAL_REMOVAL_RANGES = [
    ('06/13/2009', '06/25/2009'),
    ('05/10/2010', '05/14/2010'),
    ('09/17/2010', '09/28/2010'),
    ('05/06/2011', '05/12/2011'),
    ('06/21/2011', '06/27/2011'),
    ('06/06/2011', '06/13/2011'),
    ('06/27/2012', '08/10/2012'),
    ('08/03/2015', '08/12/2015'),
    ('05/25/2016', '06/27/2016'),
    ('06/29/2019', '07/15/2019'),
    ('09/11/2021', '10/05/2021'),
    ('03/22/2022', '03/31/2022')
]

class Data_Generator:
    def __init__(self):
        self.manual_selected_removal = False
        self.spike_test = False
        self.roc_test = False
        self.flat_line_test = False
        self.attenuated_signal_test = False

    def generate_data_set(self) -> Data_Set:
        df_sal = self.__lighthouse_full_fetch()
        df_new, df_copy, df_diff = self.__data_cleaning_procedure(df_sal)
        inputs, labels = self.__create_training_cases(df_new)
        data_training, data_validation, data_testing = self.__split_data(inputs)
        label_training, label_validation, label_testing = self.__split_data(labels)
        return Data_Set(data_training, label_training, data_validation, label_validation, data_testing, label_testing, data_training[0].shape)
    
    def generate_data_frame(self) -> list[pd.DataFrame]:
        df_sal = self.__lighthouse_full_fetch()
        return self.__data_cleaning_procedure(df_sal)
    
    def generate_binarized_set(self) -> list[pd.DataFrame]:
        df_sal = self.__lighthouse_full_fetch()
        df_new, df_copy, df_diff = self.__data_cleaning_procedure(df_sal)
        inputs, labels = self.__create_binarized_cases(df_copy, df_diff)
        data_training, data_validation, data_testing = self.__split_data(inputs)
        label_training, label_validation, label_testing = self.__split_data(labels)
        return Data_Set(data_training, label_training, data_validation, label_validation, data_testing, label_testing, data_training[0].shape)

    def __data_cleaning_procedure(self, df):

        df_copy = df.copy(deep=True)
        df_diff = df.copy(deep=True)
        if(self.manual_selected_removal):
            df = self.__manual_selected_removal(df)
        if(self.spike_test):
            df = self.__spike_test(df)
        if(self.roc_test):
            df = self.__roc_test(df)
        if(self.flat_line_test):
            df = self.__flat_line_test(df)
        if(self.attenuated_signal_test):
            df = self.__attenuated_signal_test(df)
        
        df_diff.drop(df.index, inplace=True)
        return df, df_copy, df_diff

    ## Cleaning methods
    def __manual_selected_removal(self, df):

        for removal_range in MANUAL_REMOVAL_RANGES:
            start_date = datetime.strptime(removal_range[0], '%m/%d/%Y')
            end_date = datetime.strptime(removal_range[1], '%m/%d/%Y')
            df = df.drop(df.loc[(df.index >= start_date) & (df.index <= end_date)].index)
        return df
    
    def __spike_test(self, df, colName= 'sal'):
        '''
            3_std >= |avg(t, t-1, t-2)|
        '''
        three_std = df[colName].std() * 3  
        drop_idx =[three_std >= abs(df[colName][i - 1] - ((df[colName][i - 2] + df[colName][i - 1] + df[colName][i]) / 3)) for i in range(2, len(df) - 1)] 
        drop_idx = [True] * 3 + drop_idx
        return df[drop_idx]

    def __roc_test(self, df, colName= 'sal'):
        '''
            param: Standard Deviation Scaler (SDS) : set to 3
            param: Range of standard deviation (RANGE) : set to 25 via recommendation
            |t - t-1| <= stdv(t:t-RANGE) * SDS
        '''
        drop_idx =[abs(df[colName][i] - df[colName][i - 1]) <= df[colName][i-24:i].std() * 3 for i in range(24, len(df) - 1)] 
        drop_idx = [True] * 25 + drop_idx
        return df[drop_idx]
    
    def __flat_line_test(self, df, colName= 'sal'):
        '''
            param: Amount of sames to fail (AMNT_TO_FAIL): set to 5 via report recommendation
            param: Epsilon (EPS) a threshold for clarifying fails: set to 0.05 via report recommendation
            |t - t-1| < EPS for AMNT_TO_FAIL
        '''
        AMNT_TO_FAIL = 5
        EPS = 0.05
        drop_idx = np.full((len(df)), True)
        for idx in range(AMNT_TO_FAIL, len(df) - 1):
            count = 0
            for offset in range(AMNT_TO_FAIL):
                if abs(df[colName][idx - offset] - df[colName][idx- offset - 1]) < EPS:
                    count +=1
            if count == AMNT_TO_FAIL:
                drop_idx[idx] = False

        return df[drop_idx]
    
    def __attenuated_signal_test(self, df, colName= 'sal'):
        '''
            param: Range of time to look at (TST_TIM): set to 12 via report recommendation
            param: Minimum variation  (MIN_VAR_FAIL) a threshold for clarifying fails: set to 0.05 via report recommendation for C, .05 pmm seems reasonable 
            stdv(t:t-TST_TIM) < MIN_VAR_FAIL for AMNT_TO_FAIL
        '''
        TST_TIM = 12
        MIN_VAR_FAIL = 0.05
        drop_idx = np.full((len(df)), True)
        for idx in range(TST_TIM, len(df) - 1):
            if df[colName][idx - TST_TIM:idx].std() < MIN_VAR_FAIL:
                drop_idx[idx] = False
        return df[drop_idx]
    



    ## Original data processing!
    def __api_request(self, url: str) -> None | dict:
        """Given a url, this function attempts to hit this URL and download the response as a JSON.
        NOTE On a bad api param, throws urlib HTTPError, code 400
         :param url: str - The url to hit
         :returns data: json-like dict - the data that was downloaded
        """

        try: #Attempt download
            with urlopen(url) as response:
                data = json.loads(''.join([line.decode() for line in response.readlines()])) #Download and parse

        except HTTPError as err:
            print(f'Fetch failed, HTTPError of code: {err.status} for: {err.reason}')
            return None
        except Exception as ex:
            print(f'Fetch failed, unhandled exceptions: {ex}')
            return None
        return data


    def __lighthouse_full_fetch(self, start_time = datetime.strptime('01/01/1999', '%d/%m/%Y'), end_time = datetime.strptime('01/01/2024', '%d/%m/%Y'), series= 'sal', location_code= '072'):

        from_string = start_time.strftime('%m/%d/%y').replace('/', '%2F')
        to_string = end_time.strftime('%m/%d/%y').replace('/', '%2F')


        url = f'http://lighthouse.tamucc.edu/pd?stnlist={location_code}&serlist={series}&when={from_string}%2C{to_string}&whentz=UTC0&-action=app_json&unit=metric&elev=stnd'
        response_dict = self.__api_request(url)

        stripped_data = response_dict[location_code]['data'][series]

        datetimes = np.array([datetime.fromtimestamp(data[0] /1000) for data in stripped_data])
        values = np.array([float(data[1] if data[1] != None else np.NaN) for data in stripped_data])

        df = pd.DataFrame(
            data=values[:],
            index=datetimes[:],
            columns= [series]
        )
        return df
    
    def __create_training_cases(self, df_sal):

        data = []
        labels = []
        for window in df_sal.rolling(36, 36):

            if window.shape != (36, 1):
                continue

            reshaped_window = window.to_numpy().reshape(36)

            if np.isnan(reshaped_window).any():
                continue


            label = reshaped_window[-1]
            salinity = reshaped_window[0:24]



            labels.append(np.array([label]))
            data.append(salinity.tolist() )


        return np.array(data), np.array(labels)
    
    def __create_binarized_cases(self, df_before, df_diff):

        ones = np.full((len(df_before)), 1)


        for idx, dt in enumerate(df_before.index):
            if dt in df_diff.index:
                ones[idx] = 0
        df_before.insert(1, "label", ones, True)

        data = []
        labels = []
        for window in df_before.rolling(48, 48):

            if window.shape != (48, 2):
                continue

            if np.isnan(window.to_numpy()).any():
                continue
            
            label = window['label'][-1]
            salinity = window['sal'].to_numpy()

            labels.append(np.array([label]))
            data.append(salinity)

        return np.array(data), np.array(labels)


    
    def __split_data(self, data):
        total_size = data.shape[0]
        seventy_percent = int(total_size * 0.7)
        fifteen_percent = int(total_size * 0.15)
        first_part = data[:seventy_percent]
        second_part = data[seventy_percent:seventy_percent + fifteen_percent]
        third_part = data[seventy_percent + fifteen_percent:]
        return first_part, second_part, third_part
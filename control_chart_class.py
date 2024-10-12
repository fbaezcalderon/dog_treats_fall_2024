
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Control_chart():
    def get_labels(self,data):
        observation_labels = []

        for col in data.columns:
            if col[0] =='x':
                observation_labels.append(col)
        return observation_labels
    
    def calculate_mean(self,data,kind,observation_labels): # mean and range for each sample 
        data['R']= data.loc[:,observation_labels].max(axis=1) \
                - data.loc[:,observation_labels].min(axis=1)

        if kind =='x_mean':
            data['x_mean'] = data.loc[:,observation_labels].mean(axis=1)
        
        return data,data['R'].mean()
    
    def calculate_centre_line(self, data, kind):
        if kind == 'x_mean':
            return data['x_mean'].mean()
        if kind =='R':
            return data['R'].mean()
    
    def calculate_control_limits(self,data,range_mean,sample_size,kind):
        factors_df = pd.read_csv('factors.csv')
        
        A2 = factors_df.loc[factors_df['sample_size']==sample_size]['A2'].values[0]
        D4 = factors_df.loc[factors_df['sample_size']==sample_size]['D4'].values[0]
        D3 = factors_df.loc[factors_df['sample_size']==sample_size]['D3'].values[0]

        if kind =='R':
            return D4*range_mean,D3*range_mean
                    # UCL        ,   LCL   RANGE 
        if kind =='x_mean':


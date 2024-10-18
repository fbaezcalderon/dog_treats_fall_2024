
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Control_chart():
    def _get_labels_and_sample_size(self,data):
        observation_labels = []

        for col in data.columns:
            if col[0] =='x':
                observation_labels.append(col)
        return observation_labels,len(observation_labels)
    

    def _calculate_mean(self,data,kind,observation_labels): # mean and range for each sample 
        data['R']=np.NAN
        data['R']=data.loc[:,observation_labels].max(axis=1) - \
                        data.loc[:,observation_labels].min(axis=1)
        if kind == 'x_mean':
            data['x_mean']=data.loc[:,observation_labels].mean(axis=1) 
        
        return data,data['R'].mean()
    
    def _calculate_centre_line(self, data, kind):
        if kind == 'x_mean':
            return data['x_mean'].mean()
        if kind =='R':
            return data['R'].mean()
    
    def _calculate_control_limits(self,data,range_mean,sample_size,kind):
        factors_df = pd.read_csv('factors.csv')
        
        centre_line = self._calculate_centre_line(data=data,kind=kind) 
        A2 = factors_df.loc[factors_df['sample_size']==sample_size]['A2'].values[0]
        D4 = factors_df.loc[factors_df['sample_size']==sample_size]['D4'].values[0]
        D3 = factors_df.loc[factors_df['sample_size']==sample_size]['D3'].values[0]

        if kind =='R':
            return centre_line,D4*range_mean,D3*range_mean
                    # UCL        ,   LCL   RANGE 
        if kind =='x_mean':
            return centre_line, centre_line+A2*range_mean, centre_line-A2*range_mean

        # add more control limits for different control charts
    
    def _rule1_validator(self,ucl,lcl,data,kind):
        data['Rule 1']=np.NaN  # create a blank kind  
        for index,row in data.iterrows():  # find for observation outside control limits
            if row[kind] > ucl:   
                data.loc[index,['Rule 1']]=row[kind]
            elif row[kind] < lcl :  
                data.loc[index,['Rule 1']] =row[kind]
        return data
    
    ############################################################
    # TO-DO. Develop rule 2 validator as part of your project
    ############################################################
    def _rule3_validator(self,centre_line,ucl,lcl,data,kind):
        data['Rule 3']=np.nan
        one_sd = max((ucl-centre_line) / 3,(centre_line-lcl) / 3)
        ucl_1sd = centre_line + one_sd 
        lcl_1sd = centre_line - one_sd
        for i in range(data.shape[0]):
            segment = data.loc[i:i+4][kind]
            try:
                if (segment > ucl_1sd).value_counts()[True] >=4:
                    data.loc[ (data.index >=i) & (data.index<=i+4) & ((data[kind]>ucl_1sd)),'Rule 3']=data[kind]  
            
            except:
                pass        
            try:
                if (segment < lcl_1sd).value_counts()[True] >=4:
                    data.loc[ (data.index >=i) & (data.index<=i+4) & (data[kind] < lcl_1sd),'Rule 3']=data[kind] 
            except:
                pass
        return data, ucl_1sd,lcl_1sd
    
    ############################################################
    # TO-DO. Develop rule 4 validator as part of your project
    ############################################################

    def _rule5_validator(self,data,kind):
        data['Rule 5']=np.nan
        current_check='increasing'
        i=1
        prev_value=np.NaN
        for index, row in data.iterrows():
            if index>0:
                if row[kind] > data.loc[index-1][kind]:
                    if current_check=='increasing':
                        i=i+1
                    else:
                        i=1
                    current_check='increasing'
                    
                elif row[kind] < data.loc[index-1][kind]:
                    if current_check=='decreasing':
                        i=i+1
                    else:
                        i=1
                    current_check='decreasing'
                else:
                    i=1
                if i>=5:
                        data.loc[index-i:index,'Rule 5']=data.loc[index-i:index][kind]
        return data
    
  

    

    def _create_chart(self,data, centre_line,ucl,lcl, kind,rules):
        g=sns.FacetGrid(data,height=6, aspect=3)  # Create a facegrid
        g=g.map(sns.lineplot,'sample_no',kind)  
        g.map(sns.scatterplot,'sample_no',kind)
        plt.xticks(data['sample_no'])
        x1 = np.linspace(1,len(data)+1, 50) # create an array of 50 numbers starting from critical value and ending at 4
        y1 = np.linspace(lcl,lcl, 50) # create an array of 50 numbers starting from critical value and ending at 4
        y2 = np.linspace(ucl,ucl, 50) # create an array of 50 numbers starting from critical value and ending at 4
        
        plt.fill_between(x1,y1,y2,color='skyblue',alpha=0.25) # Fill 
        g.refline(y=centre_line,color='orange')
        plt.annotate('CL',(len(data),centre_line+centre_line*0.009),size=15)
        g.refline(y=ucl,color='green')
        plt.annotate('UCL',(len(data),ucl+ucl*0.009),size=15)
        g.refline(y=lcl,color='green')
        plt.annotate('LCL',(len(data),lcl+lcl*0.009),size=15)
        
        if 'Rule 1' in rules:
            data=self._rule1_validator(ucl,lcl,data,kind)
            g.map(sns.scatterplot,'sample_no','Rule 1',color='r',marker='>',s=100) #Plot Rule 1 
            ## https://matplotlib.org/stable/api/markers_api.html  for Markers 

        ## TO-DO 
        ##  Develop rule 2 as part of your project

        if 'Rule 3' in rules:
            data,ucl_1sd,lcl_1sd=self._rule3_validator(centre_line,ucl,lcl,data,kind)
        
            g.map(sns.scatterplot,'sample_no','Rule 3',color='r',marker='s',s=80) #Plot Rule 1
            g.refline(y=ucl_1sd, color='grey')
            g.refline(y=lcl_1sd, color='grey')

        ## TO-DO 
        ##  Develop rule 4 as part of your project

        if 'Rule 5' in rules:
            data = self._rule5_validator(data,kind)
            g.map(sns.scatterplot,'sample_no','Rule 5',color='r',marker='*',s=80) #Plot Rule 1

        g.add_legend()  
        return g 

    def plot_control_char(self,data,kind,rules=[]):
        observations_labels,n= self._get_labels_and_sample_size(data)
        data,range_mean = self._calculate_mean(data,kind,observations_labels)
        centre_line,ucl,lcl = self._calculate_control_limits(data,range_mean, n, kind)
        g=self._create_chart(data,centre_line,ucl,lcl,kind,rules)
        
        return g

# main libraries
import pandas as pd
import numpy as np
import time
# visual libraries
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')
# sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def initialisation(dataset):
    # Making a list of missing value types
    missing_values = ["n/a", "na", "--", "-", "NA", "Na", "aN", "NaN"]
    df = pd.read_csv(dataset, na_values = missing_values)
    print(df.head())
    return df

def validation(df):
    # Detecting different formats
    for c in df.columns:
        cnt = 0
    for row in df.loc[:,c]:
        try:
            if(df[c].dtype==int):
                int(row)
            elif(df[c].dtype==float):
                float(row)
        except ValueError:
            df.loc[cnt, c]=np.nan
        try:
            if(df[c].dtype==str):
                if(int(row) or float(row)):
                    df.loc[cnt, c]=np.nan
            elif(df[c].dtype==bool and (row not in ['TRUE','FALSE'])):
                df.loc[cnt, c]=np.nan
        except ValueError:
            pass
        cnt+=1
    return df

def missing_values(df):
    print(df.columns)
    primary = input('Enter the column names which is to be treated as primary key separated by commas: ').split(",")
    print(primary)
    if(df.isnull().any().sum()!=0):
        print('The dataset contains missing values')
        half_count = len(df) / 2
        df = df.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values
        print('The columns containing more than 50% missing values dropped')
        ch=input('Do you want to ignore entries containing missing values(y/n): ')
        if(ch=='y'):
            df = df.dropna()
            print('Entries with missing values dropped')
        else:
            # Replace using median 
            print('The null values will be filled with the most frequent value or median')
            for c in df.columns:
                if(df[c].dtype==int or df[c].dtype==float and (c not in primary)):
                    df[c] = df[c].fillna(df[c].median())
            df = df.dropna()
    return df

def drop_nonuni_col(df):
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    return df

def visualisation(df):
    kuchbhi = primary[0]
    if(len(primary)>1):
      kuchbhi = input('Choose the key primary key for which you choose to visualize data: '+primary)
    for c in df.columns:
      if(c not in primary and df[c].dtype!=object):
        df.plot(x='PID',y=c,marker='.')
    return df

def drop_useless(df):
    useless = input('Select the feature which is useless').split(',')
    for i in useless:
        df = df.drop([i],axis=1)
    return df

def correlation(df):
    # heat map of correlation of features
    correlation_matrix = df.corr()
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(correlation_matrix,vmax=0.8,square = True)
    plt.show()
    return df

def upper_case(df):
    df = df.apply(lambda x: x.astype(str).str.upper())
    return df


    


 
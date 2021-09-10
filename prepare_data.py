
# Preprocess data

import numpy as np
import pandas as pd
import random
from random import sample 
from random import seed

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



def normalize(df, target, feature_names, bounds):
    
    df_return = df.copy()

    # Makes sure target does not need scaling
    targets = np.unique(df[target].values)
    assert (len(targets == 2) and 0. in targets and 1. in targets)

    scaler = MinMaxScaler()
    X = df_return[feature_names]
    scaler.fit(X)
    df_return[feature_names] = scaler.transform(X)

    lower_bounds = scaler.transform([bounds[0]])
    upper_bounds = scaler.transform([bounds[1]])
    return scaler, df_return, (lower_bounds[0], upper_bounds[0])

def get_weights(df, target):
    ''' 
    Get feature importance of each feature
    '''
    cor = df.corr()
    cor_target = abs(cor[target])

    weights = cor_target[:-1]  # removing target WARNING ASSUMES TARGET IS LAST
    weights = weights / np.linalg.norm(weights)
    return weights.values

def balance_df(df, target):
    len_df_0, len_df_1 = len(df[df[target] == 0.]), len(df[df[target] == 1.])
    df_0 = df[df[target] == 0.].sample(min(len_df_0, len_df_1), random_state=0)
    df_1 = df[df[target] == 1.].sample(min(len_df_0, len_df_1), random_state=0)
    df = pd.concat((df_0, df_1))
    return df

def get_bounds(df, scale):
    low_bounds = df.min().values
    up_bounds = df.max().values

    # removing target WARNING ASSUMES TARGET IS LAST
    low_bounds = scale * low_bounds[:-1]
    up_bounds = scale * up_bounds[:-1]
    return [low_bounds, up_bounds]

def split_train_test_valid(df, test_num, val_num):
    df_train, df_test = train_test_split(df, test_size=test_num, shuffle=True, random_state=0)
    df_test, df_valid = train_test_split(df_test, test_size=val_num, shuffle=True, random_state=0)
    return df_train, df_test, df_valid

def get_group_information(dataset, sen_attribute, label):
    '''Divide records as protected_favor_group, protected_unfavor_group, 
    unprotected favor group unprotected unfavor group and calculate the corresbonding numbers
    '''
    p_fav = []
    n_fav = []
    p_unfav = []
    n_unfav = []
    for i in range(0,len(dataset)):
        if dataset[sen_attribute][i] == 1 and dataset[label][i] == 1:
            p_fav.append(i)
        elif dataset[sen_attribute][i] == 1 and dataset[label][i] == 0:
            n_fav.append(i)
        elif dataset[sen_attribute][i] == 0 and dataset[label][i] == 1:
            p_unfav.append(i)
        elif dataset[sen_attribute][i] == 0 and dataset[label][i] == 0:
            n_unfav.append(i)
            
    #print('positive label in the protected group = ', len(p_fav))
    #print('negative label in the protected group = ', len(n_fav))
    #print('positive label in the unprotected group = ', len(p_unfav))
    #print('negative label in the unprotected group = ', len(n_unfav))            
    return  p_fav, n_fav, p_unfav, n_unfav

def build_fair_dataset(dataset, sen_attribute, label, adjusted_num):
    '''
    adjusted_group:
    adjusted_num: 
    '''
    res = get_group_information(dataset, sen_attribute, label)
    
    selected_group = random.sample(res[0], adjusted_num)
    frame = [dataset.iloc[selected_group], dataset.iloc[res[1]],
             dataset.iloc[res[2]], dataset.iloc[res[3]]]
    dataset = pd.concat(frame)
    dataset = dataset.reset_index(drop=True)
    return dataset

def sample_function(required_sample_size, class_size, sample_set):
    '''
    required_sample_size: the number of desired samples in each class
    class_size: the number of actual samples in each class
    sample_set: where to sample
    '''
    samples = []
    if class_size >= required_sample_size:
        samples = sample(sample_set,required_sample_size)
    else:
        remainder = required_sample_size % class_size
        times = required_sample_size // class_size
        samples_2 = []
        for i in range(0,times):
            samples_2.append(sample(sample_set,class_size))
        samples_2.append(sample(sample_set,remainder))
        
        for j in range(0,len(samples_2)):
            samples = samples_2[j] + samples    
    return samples

def load_dataset(dataset, dataset_name, sen_attribute, test_num, val_num):

    df = dataset
    target = 'target'
    feature_names = df.columns.tolist()[:-1]
    
    # Compute the bounds for clipping
    bounds = get_bounds(df, 1)

    # Compute the weihts modelizing the expert's knowledge
    weights = get_weights(df, target)

    # Split df into train/test/valid
    df_train, df_test, df_valid = split_train_test_valid(df, test_num, val_num)

    # Build experimenation config
    config = {'Dataset': dataset_name,
              #'MaxIters': iteration,
              'Alpha': 0.001,
              'Lambda': 8.5,
              'sen_attribute': sen_attribute,
              'TrainData': df_train,
              'TestData': df_test,
              'ValidData': df_valid,
              #'Scaler': scaler,
              'FeatureNames': feature_names,
              'Target': target,
              'Weights': weights,
              'Bounds': bounds}
    return config    

def preprocess_bank(dataset):
    
    target = 'target'
    feature_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    # Compute the bounds for clipping
    bounds = get_bounds(dataset, 1)

    # Normalize the data
    scaler, df, bounds = normalize(dataset, target, feature_names, bounds)
    return df
    
def preprocess_credit_german(dataset):  
    
    dataset = fetch_openml(dataset)
    target = 'target'
    df = pd.DataFrame(data= np.c_[dataset['data'], dataset[target]], columns= dataset['feature_names'] + [target])  

    # Renaming target for training later
    df[target] = df[target].apply(lambda x: 0.0 if x == 'bad' or x == 0.0 else 1.0)

    # Subsetting features to keep only continuous, discrete and ordered categorical
    feature_names = ['checking_status', 'duration', 'credit_amount',
                 'savings_status','employment','installment_commitment',
                 'residence_since','age','existing_credits','num_dependents',
                 'own_telephone','foreign_worker']

    df = df[feature_names + [target]]
   
    # Compute the bounds for clipping
    bounds = get_bounds(df, 1)

    # Normalize the data
    scaler, df, bounds = normalize(df, target, feature_names, bounds)

    # change the age attribute to be 0 and 1
    df['age'][df['age'] > 0.28] = 1
    df['age'][df['age'] <= 0.28] = 0

    # build fair dataset
    df = build_fair_dataset(df, 'age', 'target', 200)
    return df
    
def preprocess_law(frac=1, scaler=True):
    '''
    ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime',
      'fam_inc', 'male', 'pass_bar', 'tier', 'racetxt']
    'racetxt' can have values {'Hispanic', 'American Indian / Alaskan Native', 'Black', 'White', 'Other', 'Asian'}
    '''
    LAW_FILE = "your address/law_data_clean.csv"
    data = pd.read_csv(LAW_FILE)

    # Switch two columns to make the target label the last column.
    cols = data.columns.tolist()
    cols = cols[:9]+cols[11:]+cols[10:11]+cols[9:10]
    data = data[cols]

    data = data.loc[data['racetxt'].isin(['White', 'Black'])]

    # categorical fields
    category_col = ['racetxt']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
     
    datamat = data.values
    datamat = datamat[datamat[:,9].argsort()]
    datamat = datamat[:int(len(datamat)/4)]

    A = np.copy(datamat[:, 9])

    target = datamat[:, -1]
    datamat = datamat[:, :-1]

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat[:, 9] = A

    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)
    datamat = np.random.permutation(datamat)
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]
    
    df = pd.DataFrame(datamat)
    df.columns = cols
    df.rename(columns = {'pass_bar':'target'}, inplace = True) 
    
    target = 'target'
    feature_names =['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime',
       'fam_inc', 'male', 'racetxt', 'tier']
    bounds = get_bounds(df, 1)
    scaler, df, bounds = normalize(df, target, feature_names, bounds)
    
    df = build_fair_dataset(df, 'racetxt', 'target', 450)
    return df

def preprocess_compas(dataset):
    
    dataset = dataset.drop(['Unnamed: 0'], axis='columns')
    dataset = dataset.drop(['race_Other'], axis='columns')
    dataset = dataset.drop(['age'], axis='columns')
    dataset['race_Caucasian'][dataset['race_Caucasian']==1]=2
    dataset[(dataset['race_African-American'] == 1) | (dataset['race_Caucasian'] == 2)]
    dataset = dataset.drop(['race_Caucasian'], axis='columns')
    
    dataset.rename(columns = {'race_African-American':'race'}, inplace = True) 
    dataset.rename(columns = {'two_year_recid':'target'}, inplace = True) 
    
    target = 'target'
    feature_names = ['decile_score', 'priors_count', 'v_decile_score',
       'sex_Female', 'sex_Male', 'age_cat_25 - 45', 'age_cat_Greater than 45',
       'age_cat_Less than 25', 'race']
    
    bounds = get_bounds(dataset)
    scaler, dataset, bounds = normalize(dataset, target, feature_names, bounds)
    
    dataset = build_fair_dataset(dataset, 'race', 'target', 1118)
    return dataset

def preprocess_titanic(dataset):
    
    dataset.rename(columns = {'Survived':'target'}, inplace = True) 
    
    target = 'target'
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
       'FamilySize', 'Title']
    
    bounds = get_bounds(dataset, 1)
    scaler, dataset, bounds = normalize(dataset, target, feature_names, bounds)
    
    p_fav, p_unfav, up_fav, up_unfav = get_group_information(dataset, 'Sex', 'target')
    p_fav_sample = sample_function(109, 233, p_fav)
    p_unfav_sample = sample_function(468, 81, p_unfav)

    frames = [dataset.iloc[p_fav_sample], dataset.iloc[p_unfav_sample], dataset.iloc[up_fav], 
              dataset.iloc[up_unfav]]
    dataset = shuffle(dataset)
    dataset = pd.concat(frames).reset_index(drop=True)
    return dataset
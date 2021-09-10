'''
This file contains code for preprocessing/loading datasets

Code for loading German Credit Risk dataset is modified from:
https://www.kaggle.com/kabure/predicting-credit-risk-model-pipeline

Code for loading UCI adult dataset is modified from:
https://github.com/jmikko/fair_ERM/blob/master/load_data.py

Code for loading COMPAS dataset is modified from:
https://github.com/mbilalzafar/fair-classification/blob/master/disparate_mistreatment/propublica_compas_data_demo/load_compas_data.py

Code for loading Bank Marketing dataset is modified from:
https://www.kaggle.com/mayurjain/ml-bank-marketing-solution

Original Law School Dataset at:
https://slack-redir.net/link?url=https%3A%2F%2Fweb.archive.org%2Fweb%2F20160611132146%2Fhttp%3A%2F%2Fwww2.law.ucla.edu%2Fsander%2FSystemic%2FData.htm
The original dataset is extremely unbalanced. We randomly remove many positive samples.
'''
import os
from random import seed, shuffle

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler




SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def load_german(frac=1, scaler=True):
    '''
    22 features and 1 target
    Job                            1000 non-null int64
    Credit amount                  1000 non-null int64
    Duration                       1000 non-null int64
    Purpose_car                    1000 non-null uint8
    Purpose_domestic appliances    1000 non-null uint8
    Purpose_education              1000 non-null uint8
    Purpose_furniture/equipment    1000 non-null uint8
    Purpose_radio/TV               1000 non-null uint8
    Purpose_repairs                1000 non-null uint8
    Purpose_vacation/others        1000 non-null uint8
    Sex_male                       1000 non-null uint8
    Housing_own                    1000 non-null uint8
    Housing_rent                   1000 non-null uint8
    Savings_moderate               1000 non-null uint8
    Savings_no_inf                 1000 non-null uint8
    Savings_quite rich             1000 non-null uint8
    Savings_rich                   1000 non-null uint8
    Check_moderate                 1000 non-null uint8
    Check_no_inf                   1000 non-null uint8
    Check_rich                     1000 non-null uint8
    Age_cat_Old                    1000 non-null uint8
    Foreign                        1000 non-null uint8
    Risk_bad                       1000 non-null uint8
    '''


    df_credit = pd.read_csv("datasets/new_german_credit_data.csv", index_col=0)


    #Let's look the Credit Amount column
    interval = (18, 25, 120)

    cats = ['Young', 'Old']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)


    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

    #Purpose to Dummies Variable
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    #Sex feature in dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    # Housing get dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    # Housing get Saving Accounts
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    # Housing get Risk
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    # Housing get Age categorical
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)


    #Excluding the missing columns
    del df_credit["Age"]
    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit['Risk_good']


    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])

    datamat = df_credit.drop('Risk_bad', 1).values
    target = df_credit["Risk_bad"].values
    A = np.copy(datamat[:, 11])

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat[:, 11] = A

    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)
    datamat = np.random.permutation(datamat)
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]

    return datamat




def load_law(frac=1, scaler=True):
    '''
    ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime',
      'fam_inc', 'male', 'pass_bar', 'tier', 'racetxt']
    'racetxt' can have values {'Hispanic', 'American Indian / Alaskan Native', 'Black', 'White', 'Other', 'Asian'}
    '''
    LAW_FILE = 'datasets/law_data_clean.csv'
    data = pd.read_csv(LAW_FILE)

    # Switch two columns to make the target label the last column.
    cols = data.columns.tolist()
    cols = cols[:9]+cols[11:]+cols[10:11]+cols[9:10]
    data = data[cols]

    data = data.loc[data['racetxt'].isin(['White', 'Black'])]

    # Here we apply discretisation on column marital_status
    # data.replace(['Hispanic', 'American Indian / Alaskan Native',
    #               'Black', 'Other', 'Asian'],
    #              ['Minority', 'Minority', 'Minority', 'Minority',
    #               'Minority'], inplace=True)


    # categorical fields
    category_col = ['racetxt']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    datamat = data.values

    # # Switch two columns to make the target label the last column.
    # temp = np.copy(datamat[:, 11])
    # datamat[:, 11] = datamat[:, 9]
    # datamat[:, 9] = temp


    datamat = datamat[datamat[:,9].argsort()]
    datamat = datamat[:int(len(datamat)/5)]


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


    return datamat



def load_adult(frac=1, scaler=True):
    '''
    :param frac: the fraction of data to return
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    dir = 'datasets'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    # train_addr = 'https://github.com/jmikko/fair_ERM/blob/master/datasets/adult/adult.data'
    # test_addr = 'https://github.com/jmikko/fair_ERM/blob/master/datasets/adult/adult.test'

    ADULT_TRAIN_FILE = dir+os.sep+'adult.data'
    ADULT_TEST_FILE = dir+os.sep+'adult.test'

    # check_data_file(ADULT_TRAIN_FILE, train_addr)
    # check_data_file(ADULT_TEST_FILE, test_addr)

    names=[
        "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
        "hours per week", "native-country", "income"]

    data = pd.read_csv(
        ADULT_TRAIN_FILE,
        names=names, skipinitialspace=True
            )

    data_test = pd.read_csv(
        ADULT_TEST_FILE,
        names=names, skipinitialspace=True
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data


    data = data[data["workclass"] != '?']
    data = data[data["occupation"] != '?']
    data = data[data["native-country"] != '?']
    data = data[data["race"] != 'Asian-Pac-Islander']
    data = data[data["race"] != 'Amer-Indian-Eskimo']
    data = data[data["race"] != 'Other']

    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)

    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation', 'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values



    target = datamat[:, -1]
    datamat = datamat[:, :-1]

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)
    datamat = np.random.permutation(datamat)
    print('The dataset is loaded...')
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]


    return datamat



def load_compas(frac=1):
    '''
    Feature list:
    'intercept', 'age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex', 'priors_count', 'c_charge_degree'
    '''
    def add_intercept(x):

        """ Add intercept to the data before linear classification """
        m,n = x.shape
        intercept = np.ones(m).reshape(m, 1) # the constant b
        return np.concatenate((intercept, x), axis = 1)

    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
    CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the decision variable


    dir = 'datasets'
    if not os.path.isdir(dir):
        os.mkdir(dir)


    COMPAS_INPUT_FILE = dir+os.sep+"compas-scores-two-years.csv"

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])



    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]



    """ Feature normalization and one hot encoding """
    # convert class label 0 to -1
    y = data[CLASS_FEATURE]

    # print("\nNumber of people recidivating within two years")
    # print(pd.Series(y).value_counts())
    # print("\n")


    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it


    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)


        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))



    """permute the data randomly"""
    perm = list(range(X.shape[0]))
    shuffle(perm)
    X = np.array([X[i] for i in perm])
    y = np.array([y[i] for i in perm])


    X = add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert(len(feature_names) == X.shape[1])
    # print("Features we will be using for classification are:", feature_names, "\n")

    datamat = np.concatenate([X, y[:, np.newaxis]], axis=1)

    # Take fraction
    cutoff = int(np.floor(len(datamat)*frac))
    datamat = datamat[:cutoff, :]
    import copy
    ref = datamat[:, 4]
    tmp = copy.deepcopy(datamat[:, 4])

    ref[tmp==0.0] = 1.0
    ref[tmp==1.0] = 0.0

    return datamat

def load_bank(frac=1):
    '''
        job                    11162 non-null int64
        default                11162 non-null int64
        housing                11162 non-null int64
        loan                   11162 non-null int64
        contact                11162 non-null int64
        month                  11162 non-null int64
        campaign               11162 non-null int64
        Middle_Aged            11162 non-null int64
        primary                11162 non-null int64
        secondary              11162 non-null int64
        tertiary               11162 non-null int64
        unknown                11162 non-null int64
        Neg_Balance            11162 non-null int64
        No_Balance             11162 non-null int64
        Pos_Balance            11162 non-null int64
        Not_Contacted          11162 non-null int64
        Contacted              11162 non-null int64
        t_min                  11162 non-null int64
        t_e_min                11162 non-null int64
        e_min                  11162 non-null int64
        pdays_not_contacted    11162 non-null int64
        months_passed          11162 non-null float64
        married                11162 non-null int64
        singles                11162 non-null int64
        divorced               11162 non-null int64
    '''
    data = pd.read_csv('datasets/bank.csv',sep=',',header='infer', skipinitialspace=True)
    print(data.info())

    data = data.drop(['day','poutcome'],axis=1)

    def binaryType_(data):

        data.deposit.replace(('yes', 'no'), (1, 0), inplace=True)
        data.default.replace(('yes','no'),(1,0),inplace=True)
        data.housing.replace(('yes','no'),(1,0),inplace=True)
        data.loan.replace(('yes','no'),(1,0),inplace=True)
        #data.marital.replace(('married','single','divorced'),(1,2,3),inplace=True)
        data.contact.replace(('telephone','cellular','unknown'),(1,2,3),inplace=True)
        data.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
        #data.education.replace(('primary','secondary','tertiary','unknown'),(1,2,3,4),inplace=True)

        return data

    data = binaryType_(data)


    def age_(data):
        data['Middle_Aged'] = 0
        data.loc[(data['age'] <= 60) & (data['age'] >= 25),'Middle_Aged'] = 1
        return data

    def campaign_(data):


        data.loc[data['campaign'] == 1,'campaign'] = 1
        data.loc[(data['campaign'] >= 2) & (data['campaign'] <= 3),'campaign'] = 2
        data.loc[data['campaign'] >= 4,'campaign'] = 3

        return data

    def duration_(data):

        data['t_min'] = 0
        data['t_e_min'] = 0
        data['e_min']=0
        data.loc[data['duration'] <= 5,'t_min'] = 1
        data.loc[(data['duration'] > 5) & (data['duration'] <= 10),'t_e_min'] = 1
        data.loc[data['duration'] > 10,'e_min'] = 1

        return data

    def pdays_(data):
        data['pdays_not_contacted'] = 0
        data['months_passed'] = 0
        data.loc[data['pdays'] == -1 ,'pdays_not_contacted'] = 1
        data['months_passed'] = data['pdays']/30
        data.loc[(data['months_passed'] >= 0) & (data['months_passed'] <=2) ,'months_passed'] = 1
        data.loc[(data['months_passed'] > 2) & (data['months_passed'] <=6),'months_passed'] = 2
        data.loc[data['months_passed'] > 6 ,'months_passed'] = 3

        return data

    def previous_(data):

        data['Not_Contacted'] = 0
        data['Contacted'] = 0
        data.loc[data['previous'] == 0 ,'Not_Contacted'] = 1
        data.loc[(data['previous'] >= 1) & (data['pdays'] <=99) ,'Contacted'] = 1
        data.loc[data['previous'] >= 100,'Contacted'] = 2

        return data

    def balance_(data):
        data['Neg_Balance'] = 0
        data['No_Balance'] = 0
        data['Pos_Balance'] = 0

        data.loc[~data['balance']<0,'Neg_Balance'] = 1
        data.loc[data['balance'] == 0,'No_Balance'] = 1
        data.loc[(data['balance'] >= 1) & (data['balance'] <= 100),'Pos_Balance'] = 1
        data.loc[(data['balance'] >= 101) & (data['balance'] <= 500),'Pos_Balance'] = 2
        data.loc[(data['balance'] >= 501) & (data['balance'] <= 2000),'Pos_Balance'] = 3
        data.loc[(data['balance'] >= 2001) & (data['balance'] <= 10000),'Pos_Balance'] = 4
        data.loc[data['balance'] >= 10001,'Pos_Balance'] = 5

        return data

    def job_(data):

        data.loc[data['job'] == "management",'job'] = 1
        data.loc[data['job'] == "technician",'job'] = 2
        data.loc[data['job'] == "entrepreneur",'job'] = 3
        data.loc[data['job'] == "blue-collar",'job'] = 4
        data.loc[data['job'] == "retired",'job'] = 5
        data.loc[data['job'] == "admin.",'job'] = 6
        data.loc[data['job'] == "services",'job'] = 7
        data.loc[data['job'] == "self-employed",'job'] = 8
        data.loc[data['job'] == "unemployed",'job'] = 9
        data.loc[data['job'] == "student",'job'] = 10
        data.loc[data['job'] == "housemaid",'job'] = 11
        data.loc[data['job'] == "unknown",'job'] = 12

        return data

    def marital_(data):

        data['married'] = 0
        data['singles'] = 0
        data['divorced'] = 0
        data.loc[data['marital'] == 'married','married'] = 1
        data.loc[data['marital'] == 'singles','singles'] = 1
        data.loc[data['marital'] == 'divorced','divorced'] = 1

        return data

    def education_(data):

        data['primary'] = 0
        data['secondary'] = 0
        data['tertiary'] = 0
        data['unknown'] = 0
        data.loc[data['education'] == 'primary','primary'] = 1
        data.loc[data['education'] == 'secondary','secondary'] = 1
        data.loc[data['education'] == 'tertiary','tertiary'] = 1
        data.loc[data['education'] == 'unknown','unknown'] = 1

        return data

    data = campaign_(data)
    data = age_(data)
    data = education_(data)
    data = balance_(data)
    data = job_(data)
    data = previous_(data)
    data = duration_(data)
    data = pdays_(data)
    data = marital_(data)

    data_y = pd.DataFrame(data['deposit'])
    data_X = data.drop(['deposit','balance','previous','pdays','age','duration','education','marital'],axis=1)

    datamat = np.concatenate([data_X.values, data_y.values], axis=1)
    datamat = np.random.permutation(datamat)
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]

    return datamat

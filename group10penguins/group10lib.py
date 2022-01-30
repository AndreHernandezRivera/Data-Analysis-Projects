#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats





def cor_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    c_list=[]
    feat_name=X.columns.tolist()
    
    for i in X.columns.tolist():
        corr=np.corrcoef( X[i],y )[0,1]
        c_list.append(corr)
    cor_list=[0 if np.isnan(i) else i for i in c_list ]
    
    cor_feature=X.iloc[:, np.argsort( np.abs(cor_list) )[-num_feats:] ].columns.tolist()
    
    cor_support= [True if i in cor_feature else False for i in feat_name]
    
    # Your code ends here
    return cor_support, cor_feature


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    X_norm=MinMaxScaler().fit_transform(X)
    chi_selector=SelectKBest(chi2,k=num_feats)
    chi_selector.fit(X_norm,y)
    
    chi_support=chi_selector.get_support()
    chi_feature=X.iloc[:,chi_support].columns.tolist()
    
    # Your code ends here
    return chi_support, chi_feature




from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    rfe=RFE(estimator=LogisticRegression(),
            n_features_to_select=num_feats,
            step=10,
            verbose=5,
            )
    rfe.fit(X_norm,y) # normalizing with MinMaxScalar
    #rfe.fit(X,y)
    rfe_support= rfe.get_support()
    rfe_feature= X.loc[:,rfe_support].columns.tolist()
    
    # Your code ends here
    return rfe_support, rfe_feature


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    emb=SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear'), # the L1 penalty relies on the solver parm = 'liblinear'
                        max_features=num_feats,
                        )
    emb.fit(X_norm,y)
    
    embedded_lr_support=emb.get_support()
    embedded_lr_feature=X.loc[:,embedded_lr_support].columns.tolist()
    
    
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    emb=SelectFromModel(RandomForestClassifier(n_estimators=100),
                        max_features=num_feats,
                        )
    emb.fit(X,y) # normalization doesn't seem necessary
    
    embedded_rf_support=emb.get_support()
    embedded_rf_feature=X.loc[:,embedded_rf_support].columns.tolist()
    
    
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature



from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)

    lgbc=LGBMClassifier(n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=32,
                        colsample_bytree=0.2,
                        reg_alpha=3,
                        reg_lambda=1,
                        min_split_gain=0.1,
                        min_child_weight=40,
                        )
    emb=SelectFromModel(lgbc,
                        max_features=num_feats,
                        )
    emb.fit(X,y) # normalization doesn't seem necessary
    
    embedded_lgbm_support=emb.get_support()
    embedded_lgbm_feature=X.loc[:,embedded_lgbm_support].columns.tolist()
    
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


def preprocess_dataset(dataset_path="big_five_scores.csv",num_feats=3 ):
    # Your code starts here (Multiple lines)
    player_df = pd.read_csv(dataset_path) 
    
    numcols = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g',]
    catcols = ['species', 'island','sex']

    player_df = player_df[numcols+catcols]

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
    features = traindf.columns
    print(features)

    traindf = traindf.dropna()

    traindf = pd.DataFrame(traindf,columns=features)

    #y = traindf['country']>=87
    y = traindf['species_Adelie']
    X = traindf.copy()
    

    del X['species_Adelie']

    #X.head()

    #len(X.columns)

    feature_name = list(X.columns)
    

    return X, y, num_feats


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    import collections
    
    results=cor_feature+chi_feature+rfe_feature+embedded_lr_feature+embedded_rf_feature+embedded_lgbm_feature
    
    best_features={} # NOT SURE IF WE ARE ALLOWED TO ALTER THE FUNCTION STRUCTURE, SO THE OUTPUT IS NOW CHANGED TO BE MORE USEFULL
    best_features['full_features']=collections.Counter(results)
    
    max_count=max( best_features['full_features'].values() )
    
    best_features['top_features']=[k for k, v in best_features['full_features'].items() if v >= max_count]
    

    #### Your Code ends here
    return best_features,X,y
    
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def model_pipeline(X,y):

    strategies = ['mean', 'median', 'most_frequent','constant']

    algos=[LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier(),svm.SVC() ]

    for model in algos:
        results =[]

        for s in strategies:
            pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model', model)])
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

            results.append(scores)
        #print(results)
        for method, accuracy in zip(strategies, results):
            print(f"MODEL: {model}, Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}   |   Max accuracy: {round(np.max(accuracy), 3)}")
    print('done')     

best_features,X,y = autoFeatureSelector(dataset_path="penguins.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print(best_features)

model_pipeline(X,y)

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
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV ### we are not using search version because we don't want to randomize the combos, we want to try EVERY SINGLE COMBO at least once
from sklearn.metrics import classification_report



from sklearn.pipeline import make_pipeline


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def model_pipeline_no_pipes(X_train, X_test, y_train, y_test):
	###
	
	strategies = ['mean', 'median', 'most_frequent','constant']

	algo=svm.SVC()
	algo=SGDClassifier()
	algo=LogisticRegression(solver='liblinear',multi_class='auto')

	algos=[ algo ]

	
	param_grid = {'C': [1,5,10]}

	X_train, X_test, y_train, y_test=np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

	X_train_copy, X_test_copy, y_train_copy, y_test_copy =X_train, X_test, y_train, y_test
	
 
	for model in algos:
		results =[] 

		for s in strategies:

			
			X_train, X_test, y_train, y_test=X_train_copy, X_test_copy, y_train_copy, y_test_copy
			
			imputer=SimpleImputer(missing_values=np.nan,strategy=s)
			imputer=imputer.fit(X_train)
			X_train=imputer.transform(X_train)
			
			imputer=imputer.fit(X_test)
			X_test=imputer.transform(X_test)
			

			grid = GridSearchCV(model, param_grid,cv=10,n_jobs=-1)   
			
			grid.fit(X_train, y_train)   
			
			best_model = grid.best_estimator_
			inferencing_model = best_model.predict(X_test)
			
			print(s)
			print(grid.best_score_)
			print(grid.best_params_)
			print('')
			
			results.append({
				'imputation': s,
				'model': f'{model}',
				'best_score': grid.best_score_,
				'best_params': grid.best_params_,
			})

	print('done')	 
	print(results)

def model_pipeline_pipes_no_scalar(X_train, X_test, y_train, y_test):
	### 
	
	strategies = ['mean', 'median', 'most_frequent','constant']



	#algos=[LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier(),svm.SVC() ]
	algo=svm.SVC()
	algo=SGDClassifier()
	algo=LogisticRegression(solver='liblinear',multi_class='auto')

	algos=[ algo ]


	for model in algos:
		results =[] 

		for s in strategies:
			pipe=Pipeline([
				('impute',SimpleImputer(strategy=s)),
				
				('model',model)				])		
			param_grid = {'model__C': [1,5,10],'impute__strategy':[s]}


			pipeline = Pipeline([ ('impute', SimpleImputer(strategy=s)) , ('model', model) ])
			
			
			
			grid = GridSearchCV(pipe, param_grid,cv=10,n_jobs=-1,verbose=3)   
			grid.fit(X_train, y_train)   
			
			best_model = grid.best_estimator_
			inferencing_model = best_model.predict(X_test)
			
			print(s)
			print(grid.best_score_)
			print(grid.best_params_)
			print('')
			
			results.append({
				'imputation': s,
				'model': f'{model}',
				'best_score': grid.best_score_,
				'best_params': grid.best_params_,
			})

	print('done')	 
	print(results)

def model_pipeline_pipes_scalar(X_train, X_test, y_train, y_test):
	### data leakage possibility - recommend against; also cant get scalar to be outputed
	
	strategies = ['mean', 'median', 'most_frequent','constant']

	scalar_list = [StandardScaler(with_mean = True),MinMaxScaler()] ### we are risking data leakage - info beyond the training influencing training!

	#algos=[LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier(),svm.SVC() ]
	algo=svm.SVC()
	algo=SGDClassifier()
	algo=LogisticRegression(solver='liblinear',multi_class='auto')

	algos=[ algo ]


	for model in algos:
		results =[] 
		
		for scalar in scalar_list:
			for s in strategies:
				pipe=Pipeline([
					('impute',SimpleImputer(strategy=s)),
					('scalar',scalar),
					
					('model',model)				])		
				param_grid = {'model__C': [1,5,10],'impute__strategy':[s]}


				pipeline = Pipeline([ ('impute', SimpleImputer(strategy=s)),('scalar', scalar) , ('model', model) ])
				
				
				
				grid = GridSearchCV(pipe, param_grid,cv=10,n_jobs=-1,verbose=3)   
				grid.fit(X_train, y_train)   
				
				best_model = grid.best_estimator_
				inferencing_model = best_model.predict(X_test)
				
				print(grid.best_score_)
				print(grid.best_params_)
				print('')
				
				results.append({

					'model': f'{model}',
					'best_score': grid.best_score_,
					'best_params': grid.best_params_,
				})

	print('done')	 
	print(results)

def model_pipeline_pipes_scalar(X_train, X_test, y_train, y_test):
	### 
	
	strategies = ['mean', 'median', 'most_frequent','constant']

	scaler = StandardScaler(with_mean = True)


	#algos=[LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier(),svm.SVC() ]
	algo=svm.SVC()
	algo=SGDClassifier()
	algo=LogisticRegression(solver='liblinear',multi_class='auto')

	algos=[ algo ]


	for model in algos:
		results =[] 

		for s in strategies:
			pipe=Pipeline([
				('impute',SimpleImputer(strategy=s)),
				('scalar',scaler),
				
				('model',model)				])		
			param_grid = {'model__C': [1,5,10],'impute__strategy':[s]}


			pipeline = Pipeline([ ('impute', SimpleImputer(strategy=s)) , ('model', model) ])
			
			
			
			grid = GridSearchCV(pipe, param_grid,cv=10,n_jobs=-1,verbose=3)   
			grid.fit(X_train, y_train)   
			
			best_model = grid.best_estimator_
			inferencing_model = best_model.predict(X_test)
			
			print(s)
			print(grid.best_score_)
			print(grid.best_params_)
			print('')
			
			results.append({
				'imputation': s,
				'model': f'{model}',
				'best_score': grid.best_score_,
				'best_params': grid.best_params_,
			})

	print('done')	 
	print(results)


def model_pipeline_internet(X_train, X_test, y_train, y_test):
	############### https://medium.com/analytics-vidhya/ml-pipelines-using-scikit-learn-and-gridsearchcv-fe605a7f9e05
	#### seems like our train data is being transformed and rendered useless for other iterations

	pipe_lr = Pipeline([('imputer', SimpleImputer()),
					   ('scalar', StandardScaler(with_mean = True)),
					('clf', LogisticRegression(random_state=42))])

	pipe_dt = Pipeline([('imputer', SimpleImputer()),
					 ('scalar', StandardScaler(with_mean = True)),
					 ('model', DecisionTreeClassifier(random_state=42))])

	pipe_rf = Pipeline([('imputer', SimpleImputer()),
					   ('scalar', StandardScaler(with_mean = True)),
				('clf', RandomForestClassifier(random_state=42))])


	pipe_svm = Pipeline([('imputer', SimpleImputer()),
					   ('scalar', StandardScaler(with_mean = True)),
				('clf', svm.SVC(random_state=42))])
				
				
	num_transformer = Pipeline([('imputer',SimpleImputer(strategy='mean')),
		('scaler', StandardScaler())])
	cat_transformer = Pipeline([
		('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot',OneHotEncoder(handle_unknown='ignore'))])

	num_features = X_train.select_dtypes(include=['int64']).columns
	cat_features = X_train.select_dtypes(include=['object']).columns

	preprocessor = ColumnTransformer([
			('num', num_transformer, num_features),
			('cat', cat_transformer, cat_features)])

	pipe_rf = Pipeline([('preprocess',preprocessor),
	 ('clf', RandomForestClassifier(random_state=42))])
	 
	 
	 
	
	# Set grid search params
	param_range = [9, 10]
	param_range_fl = [1.0, 0.5]

	grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
			'clf__C': param_range_fl,
			'clf__solver': ['liblinear']}] 


	grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
			'clf__max_depth': param_range,
			'clf__min_samples_split': param_range[1:]}]

	grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
			'clf__C': param_range}]

	# Construct grid searches
	jobs = -1

	LR = GridSearchCV(estimator=pipe_lr,
				param_grid=grid_params_lr,
				scoring='accuracy',
				cv=10) 



	RF = GridSearchCV(estimator=pipe_rf,
				param_grid=grid_params_rf,
				scoring='accuracy',
				cv=10, 
				n_jobs=jobs)


	SVM = GridSearchCV(estimator=pipe_svm,
				param_grid=grid_params_svm,
				scoring='accuracy',
				cv=10,
				n_jobs=jobs)



	# List of pipelines for iterating through each of them
	grids = [LR,RF,SVM]

	# Creating a dict for our reference
	grid_dict = {0: 'Logistic Regression', 
			1: 'Random Forest',
			2: 'Support Vector Machine'}

	# Fit the grid search objects
	print('Performing model optimizations...')
	best_acc = 0.0
	best_clf = 0
	best_gs = ''
	for idx, gs in enumerate(grids):
		print('\nEstimator: %s' % grid_dict[idx])
		gs.fit(X_train, y_train)
		print('Best params are : %s' % gs.best_params_)
		# Best training data accuracy
		print('Best training accuracy: %.3f' % gs.best_score_)
		# Predict on test data with best params
		y_pred = gs.predict(X_test)
		# Test data accuracy of model with best params
		print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
		# Track best (highest test accuracy) model
		if accuracy_score(y_test, y_pred) > best_acc:
			best_acc = accuracy_score(y_test, y_pred)
			best_gs = gs
			best_clf = idx
	print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])




def model_pipeline_youtube(X_train, X_test, y_train, y_test):

	imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
	imputer=imputer.fit(X_train)
	X_train=imputer.transform(X_train)
	
	imputer=imputer.fit(X_test)
	X_test=imputer.transform(X_test)
	
	#X_train, X_test, y_train, y_test=np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)


	model_params = {
		'svm': {
			'model': svm.SVC(gamma=[1,0.1,0.01,0.001,0.0001]),
			'params' : {
				'C': [1,10,20],
				'kernel': ['rbf','linear','polynomial']
			}  
		},
		'random_forest': {
			'model': RandomForestClassifier(),
			'params' : {
				'n_estimators': [1,5,10]
			}
		},
		'logistic_regression' : {
			'model': LogisticRegression(solver='liblinear',multi_class='auto'),
			'params': {
				'C': [1,5,10]
			}
		},
		'sgd' : {
			'model': SGDClassifier(),
			'params': {
				'loss':['hinge','log','modified_huber'],
				'max_iter': [1000],
				'tol' : [1e-3] , 
				'random_state': [1]
				
			}
		}
	}

	scores = []

	for model_name, mp in model_params.items():
		clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
		clf.fit( X_train,y_train )# I am getting error here we have to put something like this.  clf.fit(iris.data, iris.target) data and target column name
		scores.append({
			'model': model_name,
			'best_score': clf.best_score_,
			'best_params': clf.best_params_
		})


def model_simple(X,y,**kwargs):
	"""
		scaling_list: a list of scalar model objects
		imputation_list: a list of simpleimputer strategy params (e.g. 'mean', 'median',...)
		model_param_pair: a DICT of key: model object, and value: its model parameters
	"""
	scaling_list= kwargs.get('scaling_list',None)
	imputation_list= kwargs.get('imputation_list',None)
	model_param_pair= kwargs.get('model_param_pair',None)

	if scaling_list:
		for scalar in scalar_list:
			scalar(X)
			scalar.transform(X)
	if imputation_list:
		for strat in imputation_list:
			imputer=SimpleImputer(missing_values=np.nan,strategy=strat)
			imputer=imputer.fit(X)
			X=imputer.transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify=y) # we are stratifying, not sure if penguin proportion in population is a factor when they were sampled in the wild...
	

	### run this fx on a loop changing preprocessing options
	# imputer=SimpleImputer(missing_values=np.nan,strategy=strat)
	# imputer=imputer.fit(X_train)
	# X_train=imputer.transform(X_train)


	# imputer=SimpleImputer(missing_values=np.nan,strategy=strat)
	# imputer=imputer.fit(X_test)
	# X_test=imputer.transform(X_test)
	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	# print(X_train)
	
	
	tuned_parameters = [
		{"kernel": ["rbf","linear"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
	]

	# grid = GridSearchCV(svm.SVC(), tuned_parameters,cv=10,n_jobs=-1, verbose=3).fit(X_train,y_train)
    
    
	grid = GridSearchCV(svm.SVC(), tuned_parameters,n_jobs=-1).fit(X_train,y_train)
   
	best_model = grid.best_estimator_
	inferencing_model = best_model.predict(X_test)
	
	print(grid.best_score_)
	print(grid.best_params_)
	print('')
	results=[]
	results.append({
		'model': 'svc',
		'best_score': grid.best_score_,
		'best_params': grid.best_params_,
	})




def model_pipeline(X,y):
	#### THE ONLY ONE THAT REALLY WORKS!!!!
	
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify=y) # we are stratifying, not sure if penguin proportion in population is a factor when they were sampled in the wild...

#model_pipeline_pipes_no_scalar(X_train, X_test, y_train, y_test)
# for s in ['mean', 'median', 'most_frequent','constant']:
	# model_simple(X_train, X_test, y_train, y_test , strat=s)
#model_pipeline(X,y)


model_simple(X,y,imputation_list = ['mean', 'median', 'most_frequent','constant'])

###################################
#from sklearn.model_selection import StratifiedKFold #https://www.kaggle.com/questions-and-answers/30560
#folds = 10
#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 999)
#Use "cv=skf.split(train, train_labels)" in GridSearchCV to feed these particular folds into search, and later you can use the same folds as long as you specify the same "random_state" number. This assumes that "train" holds your data and "train_labels" holds your data labels. If none of this is particularly important, you can simply use "cv=10" in GridSearchCV for 10-fold CV.

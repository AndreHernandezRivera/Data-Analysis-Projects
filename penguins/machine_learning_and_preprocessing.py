
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


def preprocess_dataset(dataset_path="big_five_scores.csv",num_feats=3,s='mean',scalar='minmax' ):
	# Your code starts here (Multiple lines)
	player_df = pd.read_csv(dataset_path) 
	
	numcols = ['bill_length_mm', 'bill_depth_mm',
				'flipper_length_mm', 'body_mass_g',]
	catcols = ['species', 'island','sex']

	player_df = player_df[numcols+catcols]

	traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
	features = traindf.columns
	#print(features)

	imputer=SimpleImputer(missing_values=np.nan,strategy=s)
	imputer=imputer.fit(traindf)
	traindf=imputer.transform(traindf) 


	#traindf = traindf.dropna() #### IMPUTATION DOES NOTHING
	

	

	traindf = pd.DataFrame(traindf,columns=features)

	#y = traindf['country']>=87
	y = traindf['species_Adelie']
	X = traindf.copy()

	del X['species_Adelie']
	
	if scalar=='minmax':
		scaler = MinMaxScaler()
		X_scaled = scaler.fit_transform(X)
		X=pd.DataFrame(X_scaled,columns=X.columns)
	if scalar=='standard':
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)	
		X=pd.DataFrame(X_scaled,columns=X.columns)
	if scalar=='manual':
		X['body_mass_kg']=X['body_mass_g']/1000
		del X['body_mass_g']
	
	
	#print(X.head() )

	feature_name = list(X.columns)
	
	return X, y, num_feats


def autoFeatureSelector(X, y, num_feats, methods=[]):

	
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
from sklearn import metrics


from sklearn.pipeline import make_pipeline


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer






def model_simple(X_train, X_test, y_train, y_test,**kwargs):

	model_params = {
		'svm': {
			'model': svm.SVC(),
			'params' : {
				'C': [1,10,20],
				'kernel': ['rbf','linear','poly'],
				"gamma": [1e-3, 1e-4],
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
	results=[]

	for mod in model_params.keys():
			
			
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
		#grid = GridSearchCV(model_params[mod]['model'], model_params[mod]['params'],cv=cv,n_jobs=-1,verbose=3).fit(X_train,y_train)
		grid = GridSearchCV(model_params[mod]['model'], model_params[mod]['params'],cv=cv,n_jobs=-1).fit(X_train,y_train)
		best_model = grid.best_estimator_
		
		y_pred = best_model.predict(X_test)
		
		print(mod)		
		print(metrics.confusion_matrix(y_test, y_pred))

		print(grid.best_score_)
		print(grid.best_params_)
		print('')
		results.append({
			'model': model_params[mod]['model'],
			'confusion_matrix': metrics.confusion_matrix(y_test, y_pred),
			'best_model': best_model,
			'best_score': grid.best_score_,
			'best_params': grid.best_params_,
		})
	
	return results


##########################
##########################
########################## We change the s,scalar, and feat params to vary the results
##########################
##########################
s='mean'
scalar='minmax'
feats=['island_Torgersen' , 'body_mass_g']

strategies = ['mean', 'median', 'most_frequent','constant']
scalars=['minmax','standard','manual','no scaling']
feature_list=['best',['','']]




X, y, num_feats = preprocess_dataset(dataset_path="penguins.csv",s=s,scalar=scalar) # separated to apply preprocessing methods easier

best_features,X,y = autoFeatureSelector(X, y, num_feats, methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm']) # only really useful if we want to get the 'best' features
print(best_features)

### feature selection ###

if feats=='best':
	X=X[best_features['top_features'] ]
if isinstance(feats, list):
	X=X[feats]
	
print(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify=y) # we are stratifying, not sure if penguin proportion in population is a factor when they were sampled in the wild...

results=model_simple(X_train, X_test, y_train, y_test) # check this function to adjust our hyperparameters and interpret our results list
print()
print(results)



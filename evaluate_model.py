# Compare Algorithms

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# load dataset
path = 'Dataset/train.csv'

#loading dataset
df = pd.read_csv(path, sep = ",", engine = "python")
#taking care of null values
df = df.fillna('0')

#label encoding
lb_make = LabelEncoder()

heads = df.columns
for i in range(len(df.columns)):
	if df[heads[i]].dtypes == 'O':
		df[heads[i]] = lb_make.fit_transform(df[heads[i]].astype(str))

#extracting input and output features
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
# prepare configuration for cross validation test harness

seed = 7

# prepare models

#model evaluation
models = []

models.append(('XGBoost',XGBRegressor()))
models.append(('GBR', ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.1,
	                                n_estimators=250, max_depth=3,
	                                learning_rate=.1, min_samples_leaf=9,
		                              min_samples_split=9)))

models.append(('RFR', RandomForestRegressor()))

# evaluate each model in turn

results = []
names = []

scoring = 'r2'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



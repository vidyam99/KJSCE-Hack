# Compare Algorithms
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import numpy as np
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import time

def analyze(dataset):
	dataframe = pandas.read_csv(dataset,error_bad_lines=False)
	col=len(dataframe.columns)-1
	data=dataframe.iloc[:,0:col]
	labels=dataframe.iloc[:,col-1]
	class_col=col
	array = dataframe.values
	X = array[:,0:col]
	Y = array[:,class_col]
	# prepare configuration for cross validation test harness
	seed = 7
	# prepare models
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('SVM', SVC()))
	models.append(('RFC',RandomForestClassifier()))
	models.append(('GB',GradientBoostingClassifier()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('NB', GaussianNB()))
	Models=['LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','SVC','RFC','GB','LinearDiscriminantAnalysis','GaussianNB']
	# evaluate each model in turn
	results = []
	names = []
	best=0
	sec_best=0
	scoring = 'accuracy'
	i=0
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		if cv_results.mean()>best:
			best=cv_results.mean()
			mo=i
		elif best>cv_results.mean()>sec_best:
			sec_best=cv_results.mean()
			mb=i
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		yield msg
		i+=1
	yield(f"Predicted model={Models[mo]}")
	yield(f"Second best model={Models[mb]}")
	# boxplot algorithm comparison
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	yield(plt.show())
	(trainData, testData, trainLabels, testLabels) = train_test_split(
		data, labels, test_size=0.25, random_state=42)
	ch=mo
	if ch==7 or ch==8:
		ch=0
	if ch==0:
		yield("No hyperparameters")
	else:
		if ch==1:
			#For LogisticRegression--------------------------------------------------------------->
			model = LogisticRegression(penalty='l2')
			dual=[True,False] 
			max_iter=[100,110,120,130,140]
			C = [1.0,1.5,2.0,2.5]
			params = dict(dual=dual,max_iter=max_iter,C=C)
		elif ch==2:
			#For DecisionTreeClassifier----------------------------------------------------------->
			#from sklearn.tree import DecisionTreeClassifier
			#making the instance
			model= DecisionTreeClassifier(random_state=1234)
			#Hyper Parameters Set
			params = {'max_features': ['auto', 'sqrt', 'log2'],
			'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
			'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
			'random_state':[123]}
		elif ch==3:
			#For SVC------------------------------------------------------------------------------->
			#from sklearn.svm import SVC
			model=SVC(max_iter=1500)
			#Hyper Parameters Set
			params = {'C': [6,7,8,9,10,11,12], 
			'kernel': ['linear','rbf']}
		elif ch==4:
			#For RandomForestClassifier----------------------------------------------------------->
			#making the instance
			model=RandomForestClassifier()
			#hyper parameters set
			params = {'criterion':['gini','entropy'],
			'n_estimators':[10,15,20,25,30],
			'min_samples_leaf':[1,2,3],
			'min_samples_split':[3,4,5,6,7], 
			'random_state':[123],
			'n_jobs':[-1]}
		elif ch==5:
			#FOr GradientBoostingClassifier------------------------------------------------------->
			#from sklearn.ensemble import GradientBoostingClassifier
			model=GradientBoostingClassifier()
			params = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
		elif ch==6:
			#For KNN------------------------------------------------------------------------------->
			#from sklearn.neighbors import KNeighborsClassifier
			#making the instance
			model = KNeighborsClassifier(n_jobs=-1)
			#Hyper Parameters Set
			params = {'n_neighbors':[5,6,7,8,9,10],
			'leaf_size':[1,2,3,5],
			'weights':['uniform', 'distance'],
			'algorithm':['auto', 'ball_tree','kd_tree','brute'],
			'n_jobs':[-1]}
	"""#ForLDA-------------------------------------------------------------------------------->
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

	#For GaussianNB------------------------------------------------------------------------>
	from sklearn.naive_bayes import GaussianNB"""
	import time
	random = RandomizedSearchCV(estimator=model, param_distributions=params, cv = 3, n_jobs=-1)

	start_time = time.time()
	random_result = random.fit(X, Y)
	random.predict(testData)
	# Summarize results
	yield("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
	yield("Execution time: " + str((time.time() - start_time)) + ' ms')
	kfold = KFold(n_splits=3, random_state=7)
	result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
	yield(result.mean())
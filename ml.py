# See: http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print '\n\n\n\n\n\n'

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Look at data
#print dataset.shape
#print dataset.head(20)
#print dataset.describe()
#print dataset.groupby('class').size()

# Data viz

## Univariate plots 

### Box and whisker plots
#dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.savefig('fig-boxwhisker.png')

### Historgram
#dataset.hist()
#plt.savefig('fig-histogram.png')

## Multivariate plots

### Scatter plot matrix
#scatter_matrix(dataset)
#plt.savefig('fig-scatter.png')

# ---------------------------------------
# 5: Evaluate some algorithms
# ---------------------------------------

## 5.1 Create a validation dataset
array = dataset.values
#print 'array: \n{}\n\n\n\n\n'.format(array)
"""
Output:
array: 
[[5.1 3.5 1.4 0.2 'Iris-setosa']
 [4.9 3.0 1.4 0.2 'Iris-setosa']
 [4.7 3.2 1.3 0.2 'Iris-setosa']
 [4.6 3.1 1.5 0.2 'Iris-setosa']
 [5.0 3.6 1.4 0.2 'Iris-setosa']
 ...
 [5.9 3.0 5.1 1.8 'Iris-virginica']]
"""

# Need to separate data from classification
X = array[:,0:4] # Get first four columns
#print 'X: \n{}\n\n\n\n\n'.format(X)
Y = array[:,4] # Get last column
#print 'Y: \n{}\n\n\n\n\n'.format(Y)

validation_size = 0.20 # 20% validation size
seed = 7 # Rando number

# See: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

## 5.2 Test harness
# Test options and evaluation metric
num_instances = len(X_train)
num_folds = 10
seed = 7
scoring = 'accuracy'

## 5.3: Build models
# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # See: http://scikit-learn.org/0.17/modules/generated/sklearn.cross_validation.KFold.html
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # See: http://scikit-learn.org/0.17/modules/generated/sklearn.cross_validation.cross_val_score.html
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "{0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std())
    #print msg

## 5.3: Select best model
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.savefig('fig-models.png')

# ---------------------------------------
# 6: Make predictions
# ---------------------------------------

# Make predictions on validation dataset
knn = KNeighborsClassifier() # Try changing this to SVC (I got a better result!!!)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print accuracy_score(Y_validation, predictions)
print confusion_matrix(Y_validation, predictions)
print classification_report(Y_validation, predictions)

print '\n\n--------------\n\n----DONE----'
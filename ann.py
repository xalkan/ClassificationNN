# 1 - Data Preprocessing

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data and avoiding the dummy variable trap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2 - Implementing NN

# actual NN code
import keras
from keras.models import Sequential     # sequential module initializes the nn
from keras.layers import Dense          # dense module creates the layers
from keras.layers import Dropout        # to prevent overfitting

# initialize the nn
classifier = Sequential()

# adding layers
# adds first hidden layer with input layer, uniformly distributed random weights and 
# rectified linear activation function
# hidden layer 1
classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# hidden layer 1 dropout -> p = 0 overfitting, p = 1 underfitting
classifier.add(Dropout(p = 0.1))
# hidden layer 2
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the nn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the nn to training set
#classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# 3 - Making Predictions and basic evaluation

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print ("True Negatives: ", cm[0, 0])
print ("False Negatives: ", cm[1, 0])
print ("True Positives: ", cm[1, 1])
print ("False Positives: ", cm[0, 1])
print ("Test set accuracy: ", (cm[1, 1] + cm[0, 0]) / len(y_test) )


# 4 - Evaluating, improving an tuning the model

# Evaluating the NN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # build only the nn architecture
    classifier = Sequential()
    classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# k-fold cross validation classifier to check real relevant accuracies, where we are in the
# bias-variance trade off
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 1)

# Fix for keras multiprocessing on windows issue
# ImportError: [joblib] Attempting to do parallel computing without protecting your import on a system that does not support forking. To use parallel-
# computing in a script, you must protect your main loop using "if __name__ == '__main__'". Please see the joblib documentation on Parallel for more information
# I think windows doesn't have os.fork() so child processes don't have parent's context to spawn properly

# for mac and linux, just uncomment this:
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# for windows
class CrossValScore(object):
    def __init__(self):
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print("Mean: ", mean)
        print("Variance: ", variance)

if __name__ == "__main__":
    CrossValScore()
    

# Improving the NN
# added dropout above

# Tuning the NN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    # build only the nn architecture
    classifier = Sequential()
    classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# GridSeachCV classifier for parameters tuning
classifier = KerasClassifier(build_fn = build_classifier)

# create a dict for hyperparameters that I want to tune
parameters = {'batch_size': [25, 32], 'epochs': [50, 200, 500], 'optimizer': ['adam', 'rmsprop']}
# create the grid search object
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
# fit grid search object to the training set
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

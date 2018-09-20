# Data Preprocessing

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

# Fitting classifier to the training set

# actual NN code
import keras
from keras.models import Sequential     # sequential module initializes the nn
from keras.layers import Dense          # dense module creates the layers

# initialize the nn
classifier = Sequential()

# adding layers
# adds first hidden layer with input layer, uniformly distributed random weights and 
# rectified linear activation function
# hidden layer 1
classifier.add(Dense(input_dim = 11, output_dim = 6, init = 'uniform', activation = 'relu'))
# hidden layer 2
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compiling the nn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the nn to training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

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
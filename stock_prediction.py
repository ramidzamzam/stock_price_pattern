# Artificial Neural Network
# Stock price prediction based on 0 : N - 1 of close price pattern
# By: Rami D
# Date: 3/27/20

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout


"""Load coverted dataset from prev step """

convert_data = pd.read_csv('stock_price_patterns.csv')

# Balance data labels 

# Labels split 
convert_data['pattern'].value_counts().plot(kind='bar')

# separate minority and majority classes
negative = convert_data[convert_data['pattern'] == 0]
positive = convert_data[convert_data['pattern'] == 1]     
        
# upsample minority
pos_upsampled = resample(negative,replace=True, n_samples=len(positive), random_state=27)
# combine majority and upsampled minority
upsampled = pd.concat([positive, pos_upsampled])

# After resampling
upsampled['pattern'].value_counts().plot(kind='bar')

# Select required data
data = upsampled.iloc[:, np.r_[0:14, 18]]

# Split vars
X = data.iloc[:, 0:14].values
y = data.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)    

""" Building the ANN """

# Hyperparam 
EPOCHS = 100
BATCH_SIZE = 10
ANN_UNITS =  14
LEARNING_RATE = 0.1
DECAY_RATE = LEARNING_RATE / EPOCHS
MOMENTUM = 0.8
DROPOUT = 0.2

# Loss optimizer 
sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=DECAY_RATE, nesterov=False)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = ANN_UNITS, kernel_initializer = 'uniform', activation = 'relu', input_dim = ANN_UNITS))

# Adding dropout to avoid overfitting 
classifier.add(Dropout(DROPOUT))

# Adding hidden layer
classifier.add(Dense(units = round(ANN_UNITS/2), kernel_initializer = 'uniform', activation = 'relu'))

# Adding hidden layer
classifier.add(Dense(units = round(ANN_UNITS/3), kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Describe activation range
describe = pd.DataFrame(y_pred).describe()

# Pick 50% 
tr =  pd.DataFrame(y_pred).quantile(0.5).values[0]
y_pred_activation = (y_pred > tr)    

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_activation)

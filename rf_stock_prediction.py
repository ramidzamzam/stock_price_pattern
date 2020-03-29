# Random Forest Calssification
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
from sklearn.ensemble import RandomForestClassifier
from pickle import dump

"""Load coverted dataset from prev step """

convert_data = pd.read_csv('stock_price_patterns_5M.csv')

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

# Shuffle data labels 
data = data.sample(frac=1).reset_index(drop=True)

# Split vars
X = data.iloc[:, 0:14].values
y = data.iloc[:, 14].values

# Reorder close price colmuns 0 ... N
X = pd.DataFrame(X)
X = X.reindex(sorted(X.columns), axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)    

""" Building Random Forest """

# Hyperparams
TREES_NUM = 200
MAX_DEPTH = 50
CRITERION = 'entropy'
MAX_FEATURE = 6
# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = TREES_NUM,
                                    max_depth = MAX_DEPTH,
                                    criterion = CRITERION, 
                                    max_features = MAX_FEATURE,
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
win = cm[0][0] + cm[1][1]
acc = round(win/len(y_pred) * 100, 2)
no_loss = win + cm[0][1]
no_loss_acc = round(no_loss/len(y_pred) * 100, 2)

# Save model and scaler
dump(sc, open('sc.pkl', 'wb'))
dump(classifier, open('classifier.pkl', 'wb'))


# Investigate optimal hypereparams
varbs = [50, 100, 150, 200, 250, 300]
pred_res = []
for varb in varbs:
    rf = RandomForestClassifier(n_estimators = TREES_NUM,
                                    max_depth = MAX_DEPTH,
                                    criterion = CRITERION, 
                                    max_features = MAX_FEATURE,
                                    random_state = 0, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, train_pred)
    win = cm[0][0] + cm[1][1]
    acc = round(win/len(y_pred) * 100, 2)
    pred_res.append(acc)
   
   
   
   

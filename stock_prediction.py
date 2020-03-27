# Artificial Neural Network
# Stock price prediction 

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

""" Constants """
AGG_TIME_FRAME = 14    
DATA_TRAINING_SIZE = 20973889
# Importing the dataset
dataset = pd.read_csv('historical_stock_prices.csv')

""" Data prep """
# Select used cols'
df = pd.DataFrame(dataset, columns=['ticker', 'close', 'volume', 'date']) 

# Drop rows with missing values
df.dropna(inplace = True)

# Sort data by ticker and date
df = df.sort_values(by = ['ticker','date'])
# Pick the size of the trainig dataset
df_head = df.head(DATA_TRAINING_SIZE)

# Convert dataset
# To [{'ticker': 'A', '0': 12, '1': 33, ... ,'13': 53 '_max': 66, '_min': 1.2, 'avg_volume': 863127 'pattern': 1}, ...]
tf_row = {}
convert_data = pd.DataFrame([])
cur_ticker = 'nan'
max_close = float('-inf')
min_close = float('inf')
avg_volume = float(0)
t_index = 0
_index = 0
list_len = len(df_head) - 1 
for _, row in df_head.iterrows():
    if (cur_ticker != row.ticker and cur_ticker != 'nan') or t_index >= AGG_TIME_FRAME or _index >= list_len:
        print('Create new row ...')
        if _index >= list_len:
            # Append last row
            if row.close >= max_close:
                max_close = row.close
            if row.close <= min_close:
                min_close = row.close
            avg_volume = avg_volume + row.volume
            str_index = str(t_index)
            tf_row[str_index] = row.close
            t_index = t_index + 1
        # New ticker or end of aggregation time frame
        if cur_ticker == 'nan':
            cur_ticker = row.ticker
        tf_row['_min'] = min_close
        tf_row['_max'] = max_close
        tf_row['ticker'] = cur_ticker
        tf_row['avg_volume'] = avg_volume/(t_index + 1)
        convert_data = convert_data.append(tf_row, ignore_index=True)
        # Re init values
        cur_ticker = row.ticker
        max_close = float('-inf')
        min_close = float('inf')
        t_index = 0
        _index = _index + 1
    else:
        print('Append to row ...')
        # Append to aggregation time frame row
        if row.close >= max_close:
            max_close = row.close
        if row.close <= min_close:
            min_close = row.close
        avg_volume = avg_volume + row.volume
        str_index = str(t_index)
        tf_row[str_index] = row.close
        t_index = t_index + 1
        _index = _index + 1


# Label data pattern 
# If row[i]( (max_close - min_clos) + max_close ) =< row[i + 1](max_close) pattern 1
# Eles 0
next_i = 0
for i in range(len(convert_data)):
    next_i = i + 1
    if convert_data.loc[i, 'ticker'] == convert_data.loc[next_i, 'ticker'] and next_i < len(convert_data) - 1:
        # Check pattern
        if (convert_data.loc[i, '_max'] - convert_data.loc[i, '_min']) + convert_data.loc[i, '_max'] < convert_data.loc[next_i, '_max']:
            convert_data.loc[i, 'pattern'] = 1
        else:
            convert_data.loc[i, 'pattern'] = 0
        
# Drop rows with missing values
convert_data.dropna(inplace = True)

# Save coverted dataset to disk
convert_data.to_csv('stock_price_patterns.csv')

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
EPOCHS = 60
BATCH_SIZE = 32
ANN_UNITS =  len(X_train)
LEARNING_RATE = 0.1
DECAY_RATE = LEARNING_RATE / EPOCHS
MOMENTUM = 0.8
DROPOUT = 0.25

# Loss optimizer 
sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=DECAY_RATE, nesterov=False)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = ANN_UNITS, kernel_initializer = 'uniform', activation = 'relu', input_dim = ANN_UNITS)

# Adding dropout to avoid overfitting 
classifier.add(Dropout(DROPOUT))

# Adding hidden layer
classifier.add(Dense(units = ANN_UNITS/2, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)    

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
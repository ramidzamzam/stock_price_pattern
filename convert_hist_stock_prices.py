# Convert historical stock data to array[0 : N -1] close price to ditect patterns
# By: Rami D
# Date: 3/27/20

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.utils import resample# -*- coding: utf-8 -*-

""" Constants """
AGG_TIME_FRAME = 14    
DATA_TRAINING_SIZE = 2000000 # Full data size 20973889
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

# Round up float .
df_head.round(2)

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
        tf_row['_min'] = round(min_close, 2)
        tf_row['_max'] = round(max_close, 2)
        tf_row['ticker'] = cur_ticker
        tf_row['avg_volume'] = round(avg_volume/(t_index + 1), 2)
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
        tf_row[str_index] = round(row.close, 2)
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
convert_data.to_csv('stock_price_patterns.csv', index=False)

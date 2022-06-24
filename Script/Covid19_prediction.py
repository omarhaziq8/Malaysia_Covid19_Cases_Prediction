# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:18:28 2022

@author: pc
"""

import os 
import pickle
import datetime
import pandas as pd 

from Modules_Covid19 import EDA,ModelCreation,ModelEvaluation
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#%% Statics
DATA_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
MMS_PATH = os.path.join(os.getcwd(),'mms_covid_cases.pkl')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)

#%%  EDA 
# Data Loading
# got '?' inside column cases_new, replace it with NaN
df = pd.read_csv(DATA_PATH,na_values='?')

#%% Data Inspection

# To check dtype of dataset, cases_new: object
df.info() #12 Nans inside cases_new column, 342 Nans for cluster section

df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
# As you can see the dtype for cases_new is object, we need to convert to numerical(float)
# by using pandas to numeric

# To check the statistics of dataset
df.describe().T

# To visualise cases and compares
eda = EDA()
eda.plot_graph(df)

# plt.figure()
# plt.plot(df['cases_new'])
# plt.plot(df['cases_active'])
# plt.plot(df['cases_recovered'])
# plt.legend(['cases_new','cases_active','cases_recovered'])
# plt.title('Malaysia Covid-19 Cases')
# plt.show()

# From plotting, it shows cases new and cases recovered are in a well trend
# Meanwhile cases active has a increasing and decreasing trend, (not fixed trend)

#%% Data Cleaning

# Nan Values inside cases_new to be remove by using df.interpolate
df['cases_new'] = df['cases_new'].interpolate() # removes NaN/interpolate for continous
# To check the NaN values still got or not, shows 0
df.isna().sum()
df.info()
# To check whter the Nans has been interpolate for cases_new
# From info, can see the cluster section has so many Nans, but no need to worry since 
# we dont need them as features, only need cases_new



#%% Features Selection

# no other features significant for predicting, only select cases_new 

#%% Preprocessing

mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

# save using pickle
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

x_train = []
y_train = []
win_size = 30 # To set 30 as from the question/constant/ refer to days

for i in range(win_size,np.shape(df)[0]):
    x_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)
# never perform train test split for time series data

#%% Model Development

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

mc = ModelCreation()
model = mc.simple_lstm_layer(x_train)

plot_model(model,show_layer_names=(True),show_shapes=(True))

x_train = np.expand_dims(x_train,axis=-1)

# CALLBACKS
tensorboard_callbacks = TensorBoard(log_dir=LOG_FOLDER_PATH)

hist = model.fit(x_train,y_train,batch_size=32,epochs=100,
                 callbacks=(tensorboard_callbacks))


#%% model evaluation
# To evaluate loss and mape by plotting 
hist.history.keys()

me = ModelEvaluation()
me.plot_model_evaluation(hist)

# From the graph plotting, it shows my model according to loss is balancing not really flunctuate
# but for the MAPE, it shows not normal distribution throughout 100 epochs, uneven distribution

#%% Testing Model and analysis

CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')
test_df = pd.read_csv(CSV_TEST_PATH) #test_df:100
test_df['cases_new'] = test_df['cases_new'].interpolate()
test_df = mms.transform(np.expand_dims(test_df['cases_new'].values,axis=-1))
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-(win_size+len(test_df)):] #winsize:30 + test_df:100

x_test = []
for i in range(win_size,len(con_test)):
    x_test.append(con_test[i-win_size:i,0])

x_test = np.array(x_test)

predicted = model.predict(np.expand_dims(x_test,axis=-1))



#%% Prediction Visualisation

me.plot_predicted_graph(test_df, predicted, mms)

# From plotting, the predicted is able to follow the trend from the actual cases
# They are not really flunctuate and still able to predict since they have high MAPE

#%%

from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

print('mae:', mean_absolute_error(test_df, predicted))
print('mse:', mean_squared_error(test_df, predicted))
print('mape:', mean_absolute_percentage_error(test_df, predicted))

print('mae_inverse:', mean_absolute_error(test_df_inversed, predicted_inversed))
print('mse_inverse:', mean_squared_error(test_df_inversed, predicted_inversed))
print('mape_inverse:', mean_absolute_percentage_error(test_df_inversed, predicted_inversed))

# Sir Warren New Intervention formula 
print((mean_absolute_error(test_df,predicted)/sum(abs(test_df))) *100)

#Conclusion
# From my result, mape inverse value is the same as sir formula after x100% 
# meaning my MAPE is low than 1% as here shows the MAPE is 0.14%
# From the model training and results, it can concludes that this model able 
# to predict new cases in the future after training from 30 past days dataset
# Maybe can try to increase number of epochs to shrink the MAPE value, minimise number of dropout rate
# For improvement, can include a web scarping algorithm to analyse the latest news 
# to polish up the model performance









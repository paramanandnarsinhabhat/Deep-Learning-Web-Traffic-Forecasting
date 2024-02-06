'''
# Web Traffic Forecasting

(This notebook will help you understand how to formulate and build a deep learning model for time series data.)


## Understanding the Problem Statement
In order to dynamically manage resources to run your website, you need to have an idea about the number of visitors who might arrive at your website at different points in time. So, the problem at hand is to predict the web traffic or number of sessions in the next hour based on the historical data.

## Load Dataset

Let us load the dataset first.

'''
import pandas as pd
import numpy as np

data=pd.read_csv('data/webtraffic.csv')

print(data.head())

#Check the shape of the data
print(data.shape)

## Data Exploration
import matplotlib.pyplot as plt
sessions = data['Sessions'].values

ar = np.arange(len(sessions))
plt.figure(figsize=(22,10))
plt.plot(ar, sessions,'r')
plt.show()

#first week web traffic
sample = sessions[:168]
ar = np.arange(len(sample))
plt.figure(figsize=(22,10))
plt.plot(ar, sample,'r')
plt.show()

'''
## Data Preparation

We will model the problem to predict the traffic for the next hour based on the previous week data i.e. 168 hours. Lets define a function to prepare the input and output data accordingly.

*Note: The input is a sequence of values and the output is a single value.*


'''
def prepare_data(seq,num):
  x=[]
  y=[]

  for i in range(0,(len(seq)-num),1):
    
    input_ = seq[i:i+num]
    output  = seq[i+num]
    
    x.append(input_)
    y.append(output)
    
  return np.array(x), np.array(y)

#Call the function
num=168
x,y= prepare_data(sessions,num)

print(len(x))

print(x[0])

print(y[0])

#Split the dataset into training and validation data
ind = int(0.9 * len(x))

x_tr = x[:ind]
y_tr = y[:ind]

x_val=x[ind:]
y_val=y[ind:]

#Normalize the input and output data as it speeds up the training process

from sklearn.preprocessing import StandardScaler

#normalize the inputs
x_scaler= StandardScaler()
x_tr = x_scaler.fit_transform(x_tr)
x_val= x_scaler.transform(x_val)

#reshaping the output for normalization
y_tr=y_tr.reshape(len(y_tr),1)
y_val=y_val.reshape(len(y_val),1)

#normalize the output
y_scaler=StandardScaler()
y_tr = y_scaler.fit_transform(y_tr)[:,0]
y_val = y_scaler.transform(y_val)[:,0]

print(x_tr.shape)

'''
As you can see here, the input data is a 2 dimenional array but the LSTM and Conv1D accepts only 3 dimensional inputs in the form of (no. of samples, no. of timesteps, no. of features)

So, reshaping the input data as per the model requirement

'''
#reshaping input data
x_tr= x_tr.reshape(x_tr.shape[0],x_tr.shape[1],1)
x_val= x_val.reshape(x_val.shape[0],x_val.shape[1],1)

print(x_tr.shape)

'''
Now, the data is ready for model training. 

<br>

## Model Building

Define the model architecture
'''

from keras.models import *
from keras.layers import *
from keras.callbacks import *

model= Sequential()

model.add(Conv1D(64, 3, padding='same', activation='relu',input_shape=(num,1)))

model.add(Conv1D(32, 5, padding='same', activation='relu',input_shape=(num,1)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()

model.compile(loss='mse',optimizer='adam')


mc = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


history=model.fit(x_tr, y_tr ,epochs=30, batch_size=32, validation_data=(x_val,y_val), callbacks=[mc])

model.load_weights('best_model.hdf5')

## Comparision with the baseline model
#build a simple model
def compute_moving_average(data):
  pred=[]
  for i in data:
    avg=np.sum(i)/len(i)
    pred.append(avg)
  return np.array(pred)

x_reshaped = x_val.reshape(-1,168)
y_pred = compute_moving_average(x_reshaped)

mse = np.sum ( (y_val - y_pred) **2 ) / (len(y_val))
print(mse)

'''
## Forecasting

**Steps to Follow**:

1. Intialize the array, say "data" with a weeks data
2. Predict for the next hour
3. Append the predicted value as the last element of array "data"
4. Skip the first element of array "data"
5. Repeat steps 2 to 4 for **N** iterations

Define a function which forecasts the traffic for the next hours from the previous week data.
'''


mse = model.evaluate(x_val,y_val)
print("Mean Square Error:",mse)


def forecast(x_val, no_of_pred, ind):
  predictions=[]

  #intialize the array with previous weeks data  
  temp=x_val[ind]

  for i in range(no_of_pred): 

    #predict for the next hour
    pred=model.predict(temp.reshape(1,-1,1))[0][0]
    
    #append the prediction as the last element of array
    temp = np.insert(temp,len(temp),pred)
    predictions.append(pred)

    #ignore the first element of array
    temp = temp[1:]

  return predictions

no_of_pred =24
ind=72
y_pred= forecast(x_val,no_of_pred,ind)

y_true = y_val[ind:ind+(no_of_pred)]

y_true= y_scaler.inverse_transform(y_true)
y_pred= y_scaler.inverse_transform(y_pred)

def plot(y_true,y_pred):
  ar = np.arange(len(y_true))
  plt.figure(figsize=(22,10))
  plt.plot(ar, y_true,'r')
  plt.plot(ar, y_pred,'y')
  plt.show()

plot(y_true,y_pred)





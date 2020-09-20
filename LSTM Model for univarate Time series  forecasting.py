#!/usr/bin/env python
# coding: utf-8

# In[1]:



# univariate lstm example
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


# In[5]:


# preparing independent and dependent features
def prepare_data(timeseries_data, n_features):
    X, y =[],[]
    for i in range(len(timeseries_data)):
# find the end of this pattern
            end_ix = i + n_features
# check if we are beyond the sequence
            if end_ix > len(timeseries_data)-1:
                     break
# gather input and output parts of the pattern
            seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


# In[6]:



# define input sequence
timeseries_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = prepare_data(timeseries_data, n_steps)


# In[7]:



print(X),print(y)


# In[8]:


X.shape


# In[9]:


# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# # Building LSTM Model

# In[10]:


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=1)


# In[ ]:


# demonstrate prediction for next 10 days
from array import array
x_input =array([187, 196, 210])
temp_input=list(x_input)
lst_output=[]
i=0
while(i<10):
    
    if(len(temp_input)>3):
        x_input=array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
    

print(lst_output)


# In[15]:



timeseries_data


# In[16]:



len(timeseries_data)


# In[17]:



lst_output


# In[ ]:


lst


# # Visualizaing The Output

# In[19]:


import matplotlib.pyplot as plt


# In[20]:


day_new=np.arange(1,10)
day_pred=np.arange(10,20)


# In[ ]:


plt.plot(day_new,timeseries_data)
plt.plot(day_pred,lst_output)


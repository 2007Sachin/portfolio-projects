#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"


#importing the csv file
df=pd.read_csv(r"C:\Users\sachi\Downloads\Instagram-Reach.csv")


# In[11]:


#reading the data head
df.head()


# In[44]:



# Convert date column to datetime datatype
df['Date'] = pd.to_datetime(df['Date'])
df.head()


# In[13]:


# analyze the trend of Instagram reach over time using a line chart
fig = px.line(df, x='Date', y='Instagram reach', title='Instagram reach Over Time')
fig.show()


# In[45]:


# analyze Instagram reach for each day using a bar chart
df_grouped = df.groupby('Date').sum().reset_index()

# create a bar chart of Instagram reach over time
fig = px.bar(df_grouped, x='Date', y='Instagram reach', title='Instagram reach by day')
fig.show()


# In[46]:


#create the code for the distribution of Instagram reach using a box plot

fig = px.box(df, y="Instagram reach", title="Distribution of Instagram Reach")
fig.show()


# In[47]:


#create a day column and analyze reach based on the days of the week  and write the code


df['Day of Week'] = df['Date'].dt.day_name()


# In[47]:


df.head()


# In[48]:


import numpy as np



# Convert date column to datetime datatype
df['Date'] = pd.to_datetime(df['Date'])

# Compute daily stats
day_stats = df.groupby(df['Date'].dt.day_name())['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
day_stats.rename(columns={'Date': 'day'}, inplace=True)

# Create figure
fig = go.Figure()


# In[26]:


fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['day'], y=day_stats['mean'], name='Mean'))
fig.add_trace(go.Bar(x=day_stats['day'], y=day_stats['median'], name='Median'))
fig.add_trace(go.Bar(x=day_stats['day'], y=day_stats['std'], name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', xaxis_title='Day', yaxis_title='Instagram Reach')
fig.show()



# In[49]:


#Forecasting using Time Series Forecasting
   
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = df[["Date", "Instagram reach"]]

result = seasonal_decompose(df['Instagram reach'], 
                           model='multiplicative', 
                           period=100)

fig = plt.figure()
fig = result.plot()

fig = mpl_to_plotly(fig)
fig.show()    


# In[32]:


pd.plotting.autocorrelation_plot(df["Instagram reach"])


# In[50]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df["Instagram reach"], lags = 100)


# In[ ]:


p, d, q = 8, 1, 2

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(df['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# In[43]:


predictions = model.predict(len(df), len(df)+100)

trace_train = go.Scatter(x=df.index, 
                         y=df["Instagram reach"], 
                         mode="lines", 
                         name="Training Data")
trace_pred = go.Scatter(x=predictions.index, 
                        y=predictions, 
                        mode="lines", 
                        name="Predictions")

layout = go.Layout(title="Instagram Reach Time Series and Predictions", 
                   xaxis_title="Date", 
                   yaxis_title="Instagram Reach")

fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
fig.show()


# In[ ]:





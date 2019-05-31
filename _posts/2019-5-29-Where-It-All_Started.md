---
layout: post
title: Where It All Started
---
![2019-5-29-Where-It-All-Started](/images/frangipani.png "2019-5-29-Where-It-All-Started")


this code was written 18 months ago in python 2.7!!

**Python Code Block:**

First of all, I imported necessary packages 

```python

#import necessary packages

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
import fitter
import holtwinters
import scipy
import lifetimes
import random
import json
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import skimage
import skimage.io
%matplotlib inline

#load datasets

sales = pd.read_csv('sales.csv')
products = pd.read_csv('products.csv')
line_items = pd.read_csv('line-items.csv')
customers = pd.read_csv('customers.csv')

```

give some numbers about datasets~!

```python

sales.head()
sales.dtypes
sales.info()
sales.columns
sales.customer_id.nunique()

```

**Add Month, Year, Day, Hour column into the sales dataframe**


```python

sales['sale_date'] = pd.to_datetime(sales.sale_date)
sales['local_sale_time'] =      sales['sale_date'].dt.tz_localize('UTC').dt.tz_convert('Australia/Brisbane')

#add columns into the dataframe

sales['Year'] = sales.local_sale_time.dt.year
sales['Month'] = sales.local_sale_time.dt.month
sales['Day'] = sales.local_sale_time.dt.day
sales['Hour'] = sales.local_sale_time.dt.hour

```

Instagram first step! You will need this info later

```python

#They started posting on Instagram on 2016-11-09
#Lets see the number of transactions after they started using Instagram

print 'The number of transactions after they started using Instagram:', sales.local_sale_time[sales.local_sale_time > '2016-11-09'].count()
print sales.local_sale_time[sales.local_sale_time > '2016-11-09'].max()
print sales.local_sale_time[sales.local_sale_time > '2016-11-09'].min()

```

```python

#Yearly customer transactions

yearly_cust_trans_2014 = sales.Year[sales.Year == 2014].count()
yearly_cust_trans_2015 = sales.Year[sales.Year == 2015].count()
yearly_cust_trans_2016 = sales.Year[sales.Year == 2016].count()
yearly_cust_trans_2017 = sales.Year[sales.Year == 2017].count()

print 'yearly_cust_trans_2014:', yearly_cust_trans_2014
print 'yearly_cust_trans_2015:', yearly_cust_trans_2015
print 'yearly_cust_trans_2016:', yearly_cust_trans_2016
print 'yearly_cust_trans_2017:', yearly_cust_trans_2017

```


```python

#how many transactions has each customer made??

customers_transactions = sales.groupby(['customer_id', 'local_sale_time'])[['total_price']].sum()
customers_transactions

```

**Weekly transactions**

```python

fig, ax = plt.subplots(figsize=(14,8))

sales['weekofyear'] = sales.local_sale_time.dt.weekofyear
sales.groupby(['Year', 'weekofyear']).id.count().plot(ax=ax)
plt.xlabel('Dates_weekly', fontsize = 14)
plt.ylabel('Number of transactions', fontsize = 14)
plt.title('Number of transactions per week', fontsize = 14)

```
# Cycle of transactions

**Number of transactions per week**

```python

#assigning Sunday as start of the week, create transactions_by_week

week_start = pd.date_range(dt.datetime(2014,12,1), dt.datetime(2017,8,7), freq='W')
week_finish = pd.date_range(dt.datetime(2014,12,8), dt.datetime(2017,8,14), freq='W')
transactions = []
for i in range(len(week_start)):
    transactions.append(sales[(sales.local_sale_time > week_start[i]) & 
                              (sales.local_sale_time < week_finish[i])].shape[0])
    
transactions_by_week = pd.Series(data=transactions, index=week_start)

```

```python

# Show special days in the graph (Valentine's Day, Mother's Day, Christmas)

fig, ax = plt.subplots(figsize=(14,8))
transactions_by_week.plot(ax=ax, label = 'Transactions')

ax.axvline(dt.datetime(2015,2,14), linestyle='dashed', c='purple', label = 'Valentines Day') 
ax.axvline(dt.datetime(2016,2,14), linestyle='dashed', c='purple')
ax.axvline(dt.datetime(2017,2,14), linestyle='dashed', c='purple')

ax.axvline(dt.datetime(2015,5,10), linestyle='dashed', c='pink', label = 'Mothers Day')
ax.axvline(dt.datetime(2016,5,8),  linestyle='dashed', c='pink')
ax.axvline(dt.datetime(2017,5,14), linestyle='dashed', c='pink')

ax.axvline(dt.datetime(2015,12,25), linestyle='dashed', c='green', label = 'Christmas')
ax.axvline(dt.datetime(2016,12,25), linestyle='dashed', c='green')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = 'best')
plt.xlabel('Dates_weekly', fontsize = 14)
plt.ylabel('Number of transactions', fontsize = 14)
plt.title('Number of transactions per week', fontsize = 14, fontweight = 'bold')

```

![2019-5-29-Where-It-All-Started](/images/weekly_number_of_transactions.png "2019-5-29-Where-It-All-Started")

# Time Series by HOLT-WINTERS

**Transactions value per week**

```python

#calculate the total sales by week

transactions_value = []
for i in range(len(week_start)):
    transactions_value.append(sales[(sales.local_sale_time > week_start[i]) & 
                              (sales.local_sale_time < week_finish[i])].total_price.sum())
    
transaction_value_by_week = pd.Series(data=transactions_value, index=week_start)

```

**Predictions about the number of transactions per week**

```python

#Holt-winters library doesn't understand pandas data 
#leave off some data so that you can compare how well the prediction matches reality
#the cycle length for the seasonal component
#how far ahead do I want to predict

fig, ax = plt.subplots(figsize=(18,10))

prediction_length = 72
truncate_length = prediction_length/2

predicted_values = holtwinters.multiplicative(list(transactions_by_week)[4:-truncate_length], 52, prediction_length)
prediction_timeframe = pd.date_range(
    transactions_by_week.index[-truncate_length], 
    freq='W', periods=prediction_length)
prediction_value_series = pd.Series(data=predicted_values[0],  index=prediction_timeframe)

prediction_value_series.plot(ax=ax, c='red', label = 'prediction')
transactions_by_week.plot(ax=ax, c='blue', label = 'actual')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = 'best', prop={'size': 14})

plt.xlabel('Dates_weekly', fontsize = 14)
plt.ylabel('Number of transactions', fontsize = 14)
plt.title('Predictions about the number of transactions per week', fontsize = 14, fontweight = 'bold')

```
![2019-5-29-Where-It-All-Started](/images/timeseries.png "2019-5-29-Where-It-All-Started")

```python



```

![2019-5-29-Where-It-All-Started](/images/prediction_transactions_value.png "2019-5-29-Where-It-All-Started")

![2019-5-29-Where-It-All-Started](/images/alive-dead.png "2019-5-29-Where-It-All-Started")

![2019-5-29-Where-It-All-Started](/images/sales_likes.png "2019-5-29-Where-It-All-Started")

<p align="center">

![2019-5-29-Where-It-All-Started](/images/sales_vs_likes.png "2019-5-29-Where-It-All-Started")

![2019-5-29-Where-It-All-Started](/images/648.png "2019-5-29-Where-It-All-Started")

























*The image of the Frangipani above was taken from: 
<https://www.bhg.com.au/cdnstorage/cache/6/3/6/0/8/9/x636089759e1b42fcf8a2fd3a5050a1a9923494a6.jpg.pagespeed.ic.v1AO_uIThA.jpg>

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

![2019-5-29-Where-It-All-Started](/images/number_of_transactions.png "2019-5-29-Where-It-All-Started")

# Cycle of transactions

**Number of transactions per week**

```python

#assigning Sunday as start of the week, create transactions_by_week

week_start = pd.date_range(dt.datetime(2014,12,1), dt.datetime(2017,8,7), freq='W', tz = 'Australia/Brisbane')
week_finish = pd.date_range(dt.datetime(2014,12,8), dt.datetime(2017,8,14), freq='W', tz = 'Australia/Brisbane')
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

**Transactions value per week**

```python

#calculate the total sales by week

transactions_value = []
for i in range(len(week_start)):
    transactions_value.append(sales[(sales.local_sale_time > week_start[i]) & 
                              (sales.local_sale_time < week_finish[i])].total_price.sum())
    
transaction_value_by_week = pd.Series(data=transactions_value, index=week_start)

```

![2019-5-29-Where-It-All-Started](/images/transactions_value_per_week.png "2019-5-29-Where-It-All-Started")

![2019-5-29-Where-It-All-Started](/images/timeseries.png "2019-5-29-Where-It-All-Started")

```python

f = fitter.Fitter(transactions_by_week[transactions_by_week.index < dt.datetime(2017,1,1)])
f.fit()
f.summary()

f = fitter.Fitter(transactions_by_week[transactions_by_week.index > dt.datetime(2017,1,1)])
f.fit()
f.summary()
```

```python

scipy.stats.mannwhitneyu(
 transactions_by_week[transaction_value_by_week.index < dt.datetime(2017,1,1)],
 transactions_by_week[transaction_value_by_week.index > dt.datetime(2017,1,1)]
)

```

# Customers life-time analysis


```python
import lifetimes
cust_lifetime = lifetimes.utils.summary_data_from_transaction_data(sales, 'customer_id', 'local_sale_time', 'total_price', freq='W')
cust_lifetime

cust_lifetime.monetary_value[cust_lifetime.monetary_value < 0]

cust_lifetime2 = cust_lifetime[(cust_lifetime.monetary_value != -66.0) & (cust_lifetime.monetary_value != 0.0)]
cust_lifetime2.sort_values('monetary_value', ascending = False)
```

- `frequency` represents the number of *repeat* purchases the customer has made. This means that it's one less than the total number of purchases. 
- `T` represents the age of the customer in whatever time units chosen (weekly above). This is equal to the duration between a customer's first purchase and the end of the period under study.
- `recency` represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

**Visualise the Frequency/Recency Matrix**

```python

import lifetimes.plotting
lifetimes.plotting.plot_frequency_recency_matrix(bgf, cmap = 'hot')

```
![2019-5-29-Where-It-All-Started](/images/freq_recency_mat.png "2019-5-29-Where-It-All-Started")

```python

lifetimes.plotting.plot_probability_alive_matrix(bgf, cmap = 'rainbow')

```

![2019-5-29-Where-It-All-Started](/images/prob_alive_mat.png "2019-5-29-Where-It-All-Started")

```python

t = 52
cust_lifetime['predicted_purchases'] = cust_lifetime.apply(lambda r: 
             bgf.conditional_expected_number_of_purchases_up_to_time(t, r['frequency'], r['recency'], r['T']), axis=1)

best_projected_cust = cust_lifetime.sort_values('predicted_purchases').tail(6)
best_projected_cust

```

```python

lifetimes.plotting.plot_period_transactions(bgf, figsize = (14,8))

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc = 'best', prop={'size': 20})

plt.xlabel('Number of Calibration Period Transactions', fontsize = 14)
plt.ylabel('Customers', fontsize = 14)
plt.title('Frequency of Repeat Transactions', fontsize = 14)

```

![2019-5-29-Where-It-All-Started](/images/calib_transactions.png "2019-5-29-Where-It-All-Started")

**INSTAGRAM**

text likes and photos were already scraped from Insta by the help of another script found on Github.

```python

with open('--------.json','r') as f:
    result = json.loads(f.read())

epoch = []
for i in range (len(results)):
    epoch.append(int(results[i]['created_time']))
    

dates = pd.Series(epoch).apply((lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(x)))))
dates

likes =[]
for i in range (len(results)):
    likes.append(results[i]['likes']['count'])
likes

text =[]
for i in range (len(results)):
    if results[i]['caption'] is None:
        text.append('No caption')
    else:
        text.append(results[i]['caption']['text'])

```

#Create a dataframe : two columns : text, likes

#stopwords : is, the etc

#stemming

#CountVectorizer(df.text, stop_words='english')

#Linear Regression

#Interpret the co-efficients to look at words which influence more likes.

```python

df = {'likes': likes, 'text': text}
#df = pd.DataFrame([likes, text]).T
df_insta = pd.DataFrame(df)
df_insta
df_insta.sort_values('likes', ascending = False)

```

```python

df_insta.loc[0]['text']

df_insta.text

```

```python

stop_words = set(stopwords.words('english'))
words = []
for i in range (len(df_insta.index)):
    w = nltk.word_tokenize(df_insta.loc[i]['text'])
    w = [x for x in w if x.lower() not in stop_words]
    words.append(w)
words

hash_tags = [x.lstrip("#") for x in " ".join(df_insta['text']).split() if x.startswith('#')]
total_words =  [x.lstrip("#") for x in (" ".join(df_insta['text']).split())]

```

```python

len(set(hash_tags)), len(set(total_words))

set([h for h in hash_tags if h in total_words])

set([h for h in total_words if h not in hash_tags])

import sklearn.feature_extraction.text
df_insta['text']

tfidf = sklearn.feature_extraction.text.TfidfVectorizer

tfidf = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True,
                                                       max_features=200,
                                                        ngram_range=(1,3))
vectorised = tfidf.fit_transform(df_insta['text'])
tfidf_df = pd.DataFrame(vectorised.toarray(), columns = tfidf.get_feature_names())

```

```python

import sklearn.preprocessing
scaler = sklearn.preprocessing.StandardScaler()
X_s = scaler.fit_transform(tfidf_df)

import sklearn.model_selection

gridsearch = sklearn.model_selection.GridSearchCV(
    sklearn.linear_model.Lasso(),
    param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    cv=3
)

gridsearch.fit(X_s, likes)

```

![2019-5-29-Where-It-All-Started](/images/sales_likes.png "2019-5-29-Where-It-All-Started")

![2019-5-29-Where-It-All-Started](/images/sales_vs_likes.png "2019-5-29-Where-It-All-Started")

```python

import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image")
args = {}

image = cv2.imread(args["image"])

chans = cv2.split(image)
colors = ('b', 'r', 'g')

plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

```

![2019-5-29-Where-It-All-Started](/images/648.png "2019-5-29-Where-It-All-Started")


*The image of the Frangipani above was taken from: 
<https://www.bhg.com.au/cdnstorage/cache/6/3/6/0/8/9/x636089759e1b42fcf8a2fd3a5050a1a9923494a6.jpg.pagespeed.ic.v1AO_uIThA.jpg>

---
layout: post
title: Where It All Started
---
![2019-5-29-Where-It-All-Started](/images/frangipani.png "2019-5-29-Where-It-All-Started")

In this first post of mine, I wanted to share my Capstone Project that I did at the end of the Data Science Immersive Course I took at General Assembly Sydney. Although the Computer Vision related code is just a few lines at the end, the whole shows what kind of projects I was involved in before merging with CV. Besides, this project has led me to having greater interest in the field of CV.

The below code was written  18 months ago in Python 2.7 in a Jupyter Notebook for a wholesaler florist located in Brisbane - Australia. Due to the Non-Disclosure Agreement I've signed, I may not be able to give detailed information about the dataset and the outputs of codes in some parts. Apologies in advance for the inconvenience this incompleteness may cause.

The sales dataset which was of my main interest has 32000 rows with 24 columns such as customer id, line items, sale date, total price for almost 900 unique customers over 2 and a half years of data from December 2014 to August 2017.

What this post includes is a time series analysis to predict the future of the business, customer lifetime analysis to prevent churn and analysing hashtags by Natural Language Processing and the composition of the photos posted by some basic Computer Vision techniques to see the effects of Instagram and get more attention to the business.

**Python Code Block:**

The first part of the project can be named as "the Exploratory Data Analysis" which is done to see what's in your dataset, what you can possibly get out of it and what your next steps will be to achieve your goals. For this purpose, I started off with importing necessary libraries/packages and loading datasets provided by my client which were sales, products, line items and customers data; however, I was mainly interested in sales dataset as it contained enough data for me to perform what's needed.

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
sales['local_sale_time'] =   sales['sale_date'].dt.tz_localize('UTC').dt.tz_convert('Australia/Brisbane')

#add columns into the dataframe

sales['Year'] = sales.local_sale_time.dt.year
sales['Month'] = sales.local_sale_time.dt.month
sales['Day'] = sales.local_sale_time.dt.day
sales['Hour'] = sales.local_sale_time.dt.hour

```

I also checked if there was enough data to analyse their Instagram account with hashtags and likes under the posts as I had dediced to do see how their posts affect the sales. Luckily, I had enough data there.

```python

#They started posting on Instagram on 2016-11-09
#Lets see the number of transactions after they started using Instagram

print 'The number of transactions after they started using Instagram:', sales.local_sale_time[sales.local_sale_time > '2016-11-09'].count()
print sales.local_sale_time[sales.local_sale_time > '2016-11-09'].max()
print sales.local_sale_time[sales.local_sale_time > '2016-11-09'].min()

```
And some other things:

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

#how many transactions has each customer made??

customers_transactions = sales.groupby(['customer_id', 'local_sale_time'])[['total_price']].sum()
customers_transactions

```

**Weekly transactions**

I thought it would be suitable for this business to display the number of transactions weekly as it was a wholesaler florist and the plot below was showing that there was a replicated tendency, let's call it "cycle', through the date range of the dataset with its spikes going up and down.

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

I put some vertical lines on the plot to explain those spikes. As you know, there are three main special days in a year which are Valentine's Day, Mother's Day and Christmas Day. Unfortunately, we still don't buy our fathers flowers on Father's Day in this patriarchal society. (The spikes going downwards show the christmas break where the business is closed.)

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

I plotted the same graph; but, this time for weekly transaction value.

```python

#calculate the total sales by week
transactions_value = []
for i in range(len(week_start)):
    transactions_value.append(sales[(sales.local_sale_time > week_start[i]) & 
                              (sales.local_sale_time < week_finish[i])].total_price.sum())
    
transaction_value_by_week = pd.Series(data=transactions_value, index=week_start)

```

![2019-5-29-Where-It-All-Started](/images/transactions_value_per_week.png "2019-5-29-Where-It-All-Started")

```python

f = fitter.Fitter(transactions_by_week[transactions_by_week.index < dt.datetime(2017,1,1)])
f.fit()
f.summary()

f = fitter.Fitter(transactions_by_week[transactions_by_week.index > dt.datetime(2017,1,1)])
f.fit()
f.summary()

```

By applying MannWhitney U test, I wanted to check if two different parts of the dataset were similar to each other with the null hypothesis of "There is no significant difference" and the result of the test was in favour of rejecting the null hypothesis. Therefore, I could easily say that Instagram had an effect on the sales. The reason why I used the date of 2017-1-1 instead of 2016-11-9 which was the day they started posting on Instagram was that I wanted to give social media some time (almost 2 months) to get in effect.

```python

scipy.stats.mannwhitneyu(
 transactions_by_week[transaction_value_by_week.index < dt.datetime(2017,1,1)],
 transactions_by_week[transaction_value_by_week.index > dt.datetime(2017,1,1)]
)

```

**ARIMA**

Autoregressive Integrated Moving Average Model (ARIMA) is one of the methods to predict future values/points in a series. 

I started off with checking missing values and stationarity which is an assumption meaning that the statistical properties in a time series such as mean, varience etc. are constant over time. Still, I needed to move onto another version of ARIMA as I couldn't get what I was hoping for from ARIMA.

```python

ts = pd.DataFrame(transaction_value_by_week)
ts.reset_index(inplace = True)
ts.columns = ['weekly', 'value']

#Just cropping a few lines from the year of 2014

ts.set_index(['weekly'], inplace = True)
ts2 = ts[4:]
ts2

#assigning holidays for ARIMA

holidays = ['2015-02-14', '2016-02-14', '2017-02-14', 
            '2015-12-25', '2016-12-25', '2015-05-10', '2016-05-08', '2017-05-14']
holidays = pd.to_datetime(holidays)

hlist = zip(holidays.week-1, holidays.year) 
mask = [(idx.week, idx.year) in hlist for idx in ts2.index]
ts2['holiday'] = 0
ts2.loc[mask,'holiday']=1
ts2.head()

#Check if there are any missing values

start = ts2.index.min() #get the start date from the data
end = ts2.index.max() #get the end date from the data
idx = pd.date_range(start, end, freq = 'W') #generate weekly time points for the given range
if ts2.shape[0] == len(idx):
    print 'No Missing Values'
else:
    print 'Missing Values'

#Time series Stationarity check
plt.plot(ts2.value)

#Lets do differencing to make it stationary
plt.plot(ts2.diff())

#Since the variance is not constant, taking log of the sales to make it constant over a period of time
ts2['log_value'] = np.log10(ts2['value'])

#Lets plot to see if the time series is stationary
plt.plot(ts2.log_value)

```

**SARIMAX**

ARIMA does not support seasonality, so it expects data to be not seasonal. That's why I switched to SARIMAX which has added seasonality component in the model with or without exogenous variables (holidays in this case).

There are 3 parameters to assign in this model which are,

- p: autoregressive order,
- d: difference order and
- q: moving average order.

```python

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

results_dict = {}
for p in [0, 1, 2, 3, 52]:
    for d in [0, 1, 2, 3]:
        for q in [0, 1, 2, 3]:
                try:
                    model = sm.tsa.statespace.SARIMAX(ts2.log_value, order = (p,d,q), seasonal_order=(1,0,1,12))
                    #model = ARIMA(ts2.log_value, order = (p, 1, q))
                    fit = model.fit()   
                    results_dict[str((p, q,d))] = fit.aic
                except:
                    print "Skipped",p,d,q
                    pass
                    
sorted(results_dict.items(), key = lambda x:x[1])

```
After the grid search above, I found the optimum parameters and used them to fit the model and get the results predicted by the model to plot the projection. As it can be seen in the graph below, the model was able to predict all the up and downs of the seasonality reasonably well.

```python

sarima = sm.tsa.statespace.SARIMAX(ts2.log_value, exog=ts2.holiday, order = (3, 1, 1), seasonal_order = (1, 1, 1, 24))
model_fit = sarima.fit()

ts2.index.max()

new_ts2 = pd.date_range(start=ts2.index.max(),periods=52,freq = 'W')
holidays = ['2017-12-25', '2018-02-14', '2018-05-13']
holidays = pd.to_datetime(holidays)

hlist = zip(holidays.week-1, holidays.year)

new_holidays = pd.DataFrame([1 if (idx.week, idx.year) in hlist else 0  for idx in new_ts2])

results = model_fit.get_forecast(steps = 52, exog=new_holidays)

#to numerically see the predictions ie. predicted value of weekly transactions

np.power(10,results.predicted_mean)

plt.plot(ts2.index, ts2.log_value)
idx = pd.date_range('2017-08-13', periods = 52, freq = 'W') #create additional data points for the plot

#confidence_intervals = results.conf_int(alpha = 0.05)
plt.plot(idx, results.predicted_mean)
#plt.fill_between(idx, confidence_intervals['lower log_sales'], confidence_intervals['upper log_sales'])

```

![2019-5-29-Where-It-All-Started](/images/timeseries.png "2019-5-29-Where-It-All-Started")

# Customer life-time analysis

Another component of this project was Customer Life-time Analysis. Not only do businesses have to acquire new customers for the sake of their existence in industry and financial health but retaining existing customers by predicting and preventing churn is also important to them. For this purpose, some analyses were performed and the results were displayed to act accordingly in the future.

```python

cust_lifetime = lifetimes.utils.summary_data_from_transaction_data(sales, 'customer_id', 'local_sale_time', 'total_price', freq='W')
cust_lifetime

cust_lifetime.monetary_value[cust_lifetime.monetary_value < 0]

cust_lifetime2 = cust_lifetime[(cust_lifetime.monetary_value != -66.0) & (cust_lifetime.monetary_value != 0.0)]
cust_lifetime2.sort_values('monetary_value', ascending = False)
```

**Visualise the Frequency/Recency Matrix**

- `frequency` represents the number of *repeat* purchases the customer has made. This means that it's one less than the total number of purchases. 
- `T` represents the age of the customer in whatever time units chosen (weekly above). This is equal to the duration between a customer's first purchase and the end of the period under study.
- `recency` represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

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


# coding: utf-8

# In[1]:


#匯入模組
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats


# In[2]:


# 讀取數據
data = pd.read_csv('data/electricity.csv')


# In[3]:


#樣本抽樣
import random

features = (["temperature"]*7330) + (["pressure"]*1200) + (["windspeed"]*1300)
sample_size = 1000    
sample = random.sample(features, sample_size)
for lang in set(sample):
    print(lang+"比例估計:", sample.count(lang)/sample_size)


# In[4]:


#區間估計
print("區間估計 - Temperature",'\n-------------------------------')
temperature_list = list(data["temperature"])
temperature_features = []
for x in range(10000):
    sample = np.random.choice(a=temperature_list, size=100)
    temperature_features.append(sample.mean())
print("母體平均:", sum(temperature_features)/10000.0)
sample_size = 100
sample = np.random.choice(a=temperature_features, size=sample_size)  
sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
z_critical = stats.norm.ppf(q=0.975)
print("Z分數:", z_critical)
margin_of_error = z_critical * sigma
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("信賴區間:",confidence_interval)
conf_int = stats.norm.interval(alpha=0.95, loc=sample_mean, scale=sigma)
print(conf_int[0], conf_int[1])


# In[5]:


#區間估計
print("區間估計 - Pressure",'\n-------------------------------')
pressure_list = list(data["pressure"])
pressure_features = []
for x in range(10000):
    sample = np.random.choice(a=pressure_list, size=100)
    pressure_features.append(sample.mean())
print("母體平均:", sum(pressure_features)/10000.0)
sample_size = 100
sample = np.random.choice(a=pressure_features, size=sample_size)  
sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
z_critical = stats.norm.ppf(q=0.975)
print("Z分數:", z_critical)
margin_of_error = z_critical * sigma
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("信賴區間:",confidence_interval)
conf_int = stats.norm.interval(alpha=0.95, loc=sample_mean, scale=sigma)
print(conf_int[0], conf_int[1])


# In[6]:


#區間估計
print("區間估計 - Windspeed",'\n-------------------------------')
windspeed_list = list(data["windspeed"])
windspeed_features = []
for x in range(10000):
    sample = np.random.choice(a=windspeed_list, size=100)
    windspeed_features.append(sample.mean())
print("母體平均:", sum(windspeed_features)/10000.0)
sample_size = 100
sample = np.random.choice(a=windspeed_features, size=sample_size)  
sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
z_critical = stats.norm.ppf(q=0.975)
print("Z分數:", z_critical)
margin_of_error = z_critical * sigma
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("信賴區間:",confidence_interval)
conf_int = stats.norm.interval(alpha=0.95, loc=sample_mean, scale=sigma)
print(conf_int[0], conf_int[1])


# In[7]:


#T檢定
print("T檢定 - Temperature",'\n-------------------------------')
mean = 500
sample = np.array(data["temperature"]) 
sample_size = len(sample)

sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
t_obtained = (sample_mean-mean)/sigma
print("檢定統計量:", t_obtained)
print(stats.ttest_1samp(a=sample, popmean=mean))
t_critical = stats.t.ppf(q=0.975, df=sample_size-1)
print("t分數:", t_critical)


# In[8]:


#T檢定
print("T檢定 - Pressure",'\n-------------------------------')
mean = 500
sample = np.array(data["pressure"]) 
sample_size = len(sample)

sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
t_obtained = (sample_mean-mean)/sigma
print("檢定統計量:", t_obtained)
print(stats.ttest_1samp(a=sample, popmean=mean))
t_critical = stats.t.ppf(q=0.975, df=sample_size-1)
print("t分數:", t_critical)


# In[9]:


#T檢定
print("T檢定 - Windspeed",'\n-------------------------------')
mean = 500
sample = np.array(data["windspeed"]) 
sample_size = len(sample)

sample_mean = sample.mean()
print("樣本平均:", sample_mean)
sample_stdev = sample.std()
print("樣本標準差:", sample_stdev)
sigma = sample_stdev/math.sqrt(sample_size-1)
print("樣本計算出的母體標準差:", sigma)
t_obtained = (sample_mean-mean)/sigma
print("檢定統計量:", t_obtained)
print(stats.ttest_1samp(a=sample, popmean=mean))
t_critical = stats.t.ppf(q=0.975, df=sample_size-1)
print("t分數:", t_critical)


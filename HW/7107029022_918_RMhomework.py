
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import read_csv


# In[2]:


# 讀取數據，以Date當作首欄
crimes = pd.read_csv('data/Chicago_Crimes_2012_to_2017.csv', index_col='Date')


# In[3]:


print("資料大小：",crimes.shape)
print("資料欄位：",crimes.columns)


# In[4]:


# 搜尋特定的欄位，列出所有Primary_Type(不重複)
Primary_Type_Cols = crimes.drop_duplicates(subset=['Primary Type'],keep='first')
print(Primary_Type_Cols['Primary Type'])
print("Primary Type資料大小：",Primary_Type_Cols.shape)


# In[5]:


# 印出crimes的資料類型
print(type(crimes))


# In[6]:


# 從dataframe中選擇想要的元素
crimes = crimes.iloc[:, 3: ]

#印出前5個數據
crimes.head()


# In[7]:


crimes.index = pd.to_datetime(crimes.index)


# In[8]:


# 挑選欄位名稱為Primary Type(會有重複)
s = crimes[['Primary Type']]
s.head()


# In[9]:


# 根據Primary Type每一項進行統計，統計之結果放置在新欄位counts，並以遞減排序方式呈現。
crime_count = pd.DataFrame(s.groupby('Primary Type').size().sort_values(ascending=False).rename('counts').reset_index())
crime_count.head(12)


# In[10]:


# crime_count資料大小
print("crime_count資料大小",crime_count.shape)


# In[11]:


# 匯入繪圖工具
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


# 指定主題的風格參數
sns.set(style="whitegrid",context='notebook')

# 設置圖表大小
f, ax = plt.subplots(figsize=(6, 10))


# 繪製長條圖(barplot)
sns.set_color_codes("pastel")
# 設定x軸變量為counts,y軸變量為Primary Type。
sns.barplot(x="counts", y="Primary Type", data=crime_count.iloc[:10, :], label="Total", color="b")

# 新增圖例和座標軸
# frameon:是否繪製圖像邊緣
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Type", xlabel="Crimes")
sns.despine(left=True, bottom=True)

# 顯示圖表
plt.tight_layout()
plt.show()


# In[13]:


# 經度
x = crimes['Longitude']
# 緯度
y = crimes['Latitude']
plt.title('Crimes of Longitude and Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.scatter(x,y)
plt.show()


# In[16]:


# 每年
crimes_2012 = crimes.loc[crimes['Year'] == 2012]
crimes_2013 = crimes.loc[crimes['Year'] == 2013]
crimes_2014 = crimes.loc[crimes['Year'] == 2014]
crimes_2015 = crimes.loc[crimes['Year'] == 2015]
crimes_2016 = crimes.loc[crimes['Year'] == 2016]
crimes_2017 = crimes.loc[crimes['Year'] == 2017]
#print(crimes_2012)

# 每年犯罪被arresrt的
arrest_yearly = crimes[crimes['Arrest'] == True]['Arrest']


# In[17]:


print(arrest_yearly.head())


# In[30]:


plt.subplot()

# resample:在給定的時間單位内重取樣
# A:year / M:month / W:week / D:day
# yearly arrest
arrest_yearly.resample('A').sum().plot()
plt.title('Yearly arrests')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Monthly arrest
arrest_yearly.resample('M').sum().plot()
plt.title('Monthly arrests')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Weekly arrest
arrest_yearly.resample('W').sum().plot()
plt.title('Weekly arrests')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# daily arrest
arrest_yearly.resample('D').sum().plot()
plt.title('Daily arrests')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()


# In[31]:


#Domestic violence家庭暴力分析
domestic_yearly = crimes[crimes['Domestic'] == True]['Domestic']
print(domestic_yearly.head())


# In[32]:


plt.subplot()

# yearly domestic violence
domestic_yearly.resample('A').sum().plot()
plt.title('Yearly domestic violence')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Monthly domestic violence
domestic_yearly.resample('M').sum().plot()
plt.title('Monthly domestic violence')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# Weekly domestic violence
domestic_yearly.resample('W').sum().plot()
plt.title('Weekly domestic violence')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()

# daily domestic violence
domestic_yearly.resample('D').sum().plot()
plt.title('Daily domestic violence')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
plt.show()


# In[75]:


# 分析近五年來的犯罪趨勢
theft_2012 = pd.DataFrame(crimes_2012[crimes_2012['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])
theft_2013 = pd.DataFrame(crimes_2013[crimes_2013['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])
theft_2014 = pd.DataFrame(crimes_2014[crimes_2014['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])
theft_2015 = pd.DataFrame(crimes_2015[crimes_2015['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])
theft_2016 = pd.DataFrame(crimes_2016[crimes_2016['Primary Type'].isin(['THEFT','BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT'])]['Primary Type'])


# In[76]:


# Monthly
grouper_2012 = theft_2012.groupby([pd.TimeGrouper('M'), 'Primary Type'])
grouper_2013 = theft_2013.groupby([pd.TimeGrouper('M'), 'Primary Type'])
grouper_2014 = theft_2014.groupby([pd.TimeGrouper('M'), 'Primary Type'])
grouper_2015 = theft_2015.groupby([pd.TimeGrouper('M'), 'Primary Type'])
grouper_2016 = theft_2016.groupby([pd.TimeGrouper('M'), 'Primary Type'])

data_2012 = grouper_2012['Primary Type'].count().unstack()
data_2013 = grouper_2013['Primary Type'].count().unstack()
data_2014 = grouper_2014['Primary Type'].count().unstack()
data_2015 = grouper_2015['Primary Type'].count().unstack()
data_2016 = grouper_2016['Primary Type'].count().unstack()


# In[77]:


data_2012.plot(figsize=(10, 8),linewidth=3.0)
plt.title("Top 5 monthly crimes in 2012",fontsize=20)
plt.xlabel('Months',fontsize=18)
plt.ylabel('Counts',fontsize=18)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# In[78]:


data_2016.plot(figsize=(10, 8),linewidth=3.0)
plt.title("Top 5 Monthly crimes in 2016",fontsize=20)
plt.xlabel('Months',fontsize=18)
plt.ylabel('Counts',fontsize=18)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# In[79]:


# Weekly
grouper_2012 = theft_2012.groupby([pd.TimeGrouper('W'), 'Primary Type'])
grouper_2013 = theft_2013.groupby([pd.TimeGrouper('W'), 'Primary Type'])
grouper_2014 = theft_2014.groupby([pd.TimeGrouper('W'), 'Primary Type'])
grouper_2015 = theft_2015.groupby([pd.TimeGrouper('W'), 'Primary Type'])
grouper_2016 = theft_2016.groupby([pd.TimeGrouper('W'), 'Primary Type'])

data_2012 = grouper_2012['Primary Type'].count().unstack()
data_2013 = grouper_2013['Primary Type'].count().unstack()
data_2014 = grouper_2014['Primary Type'].count().unstack()
data_2015 = grouper_2015['Primary Type'].count().unstack()
data_2016 = grouper_2016['Primary Type'].count().unstack()


# In[80]:


data_2012.plot(figsize=(10, 8),linewidth=3.0)
plt.title("Top 5 Weekly crimes in 2012",fontsize=20)
plt.xlabel('Months',fontsize=18)
plt.ylabel('Counts',fontsize=18)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


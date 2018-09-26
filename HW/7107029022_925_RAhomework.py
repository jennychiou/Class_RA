
# coding: utf-8

# In[1]:


#匯入模組
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# 讀取數據
data = pd.read_csv('data/electricity.csv')


# In[3]:


#描述性分析
s = data[['temperature','pressure', 'windspeed','electricity_consumption']]
s.describe()


# In[4]:


#散佈圖
sns.set(style='whitegrid', context='notebook')
cols = ['temperature', 'pressure', 'windspeed', 'electricity_consumption']
sns.pairplot(data[cols], size=2.5);
plt.tight_layout()
plt.show()


# In[5]:


#根據用電量分類
def get_consumption_category(wt):
    if wt < 200:
        return "<200kWh"
    elif 200 < wt < 400:
        return "200kWh~400kWh"
    elif 400 < wt < 600:
        return "400kWh~600kWh"
    elif 600 < wt < 800:
        return "600kWh~800kWh"
    elif 800 < wt < 1000:
        return "200kWh~400kWh"
    elif 1000 < wt < 1200:
        return "200kWh~400kWh"
    else:
        return ">1200kWh"
  
data["electricity_consumption_category"] = data["electricity_consumption"].map(get_consumption_category)


# In[6]:


#印出新的data
data


# In[7]:


#散佈圖加上用電量分類
cols = ['temperature', 'pressure', 'windspeed', 'electricity_consumption', 'electricity_consumption_category']
sns.pairplot(data[cols], size=2, hue="electricity_consumption_category");
plt.tight_layout()
plt.show()


# In[8]:


#相關係數
temperature = np.array(data['temperature'])
pressure = np.array(data['pressure'])
n = len(temperature)
temperature_mean = temperature.mean()
pressure_mean = pressure.mean()

diff = (temperature-temperature_mean)*(pressure-pressure_mean)
covar = diff.sum()/n
print("temperature & pressure共變異數:", covar)

corr = covar/(temperature.std()*pressure.std())
print("temperature & pressure相關係數:", corr)

df = pd.DataFrame({"temperature":temperature,"pressure":pressure})
print(df.corr(),'\n','--------------------------------------------------------------------------------------------')

pressure = np.array(data['pressure'])
windspeed = np.array(data['windspeed'])
n = len(pressure)
pressure_mean = pressure.mean()
windspeed_mean = windspeed.mean()

diff = (pressure-pressure_mean)*(windspeed-windspeed_mean)
covar = diff.sum()/n
print("pressure & windspeed共變異數:", covar)

corr = covar/(pressure.std()*windspeed.std())
print("pressure & windspeed相關係數:", corr)

df = pd.DataFrame({"pressure":pressure,"windspeed":windspeed})
print(df.corr(),'\n','--------------------------------------------------------------------------------------------')

windspeed = np.array(data['windspeed'])
electricity_consumption = np.array(data['electricity_consumption'])
n = len(windspeed)
windspeed_mean = windspeed.mean()
electricity_consumption_mean = electricity_consumption.mean()

diff = (windspeed-windspeed_mean)*(electricity_consumption-electricity_consumption_mean)
covar = diff.sum()/n
print("windspeed & electricity consumption共變異數:", covar)

corr = covar/(windspeed.std()*electricity_consumption.std())
print("windspeed & electricity consumption相關係數:", corr)

df = pd.DataFrame({"windspeed":windspeed,"electricity consumption":electricity_consumption})
print(df.corr(),'\n')


# In[9]:


#資料標準化(temperature)
from sklearn import preprocessing

temperature = np.array(data['temperature'])
pressure = np.array(data['pressure'])

df = pd.DataFrame({"temperature" : temperature, "pressure" : pressure})
print(df.head())

scaler = preprocessing.StandardScaler()
np_std = scaler.fit_transform(df)
df_std = pd.DataFrame(np_std, columns=["temperature_s", "pressure_s"])
print(df_std.head())

df_std.plot(kind="scatter", x="temperature_s", y="pressure_s")


# In[10]:


#最小最大值縮放
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
np_minmax = scaler.fit_transform(df)
df_minmax = pd.DataFrame(np_minmax, columns=["temperature_m", "pressure_m"])
print(df_minmax.head())

df_minmax.plot(kind="scatter", x="temperature_m", y="pressure_m")


# In[11]:


# 檢查是否有缺失值
data.isnull().any()


# In[12]:


#分類值
electricity_consumption_category_mapping = {">1200kWh": 6, "1000kWh~1200kWh": 5, "800kWh~1000kWh": 4,"600kWh~800kWh": 3,"400kWh~600kWh": 2,"200kWh~400kWh": 1,"<200kWh": 0}
data["electricity_consumption_category"] = data["electricity_consumption_category"].map(electricity_consumption_category_mapping)
print(data)

label_encoder = preprocessing.LabelEncoder()
data["electricity_consumption"] = label_encoder.fit_transform(data["electricity_consumption"])
data.head()


# In[13]:


#線性回歸作圖
from sklearn.linear_model import LinearRegression

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='c',s=1)
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return 

X1 = data[['temperature']].values
y1 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X1, y1)
print("Temperature VS Electricity Consumption")
print('Coefficients迴歸係數: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)
lin_regplot(X1, y1, slr)
plt.xlabel('Temperature [C]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()

X2 = data[['pressure']].values
y2 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X2, y2)
print("Pressure VS Electricity Consumption")
print('Coefficients迴歸係數: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)
lin_regplot(X2, y2, slr)
plt.xlabel('Pressure [atm]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()

X3 = data[['windspeed']].values
y3 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X3, y3)
print("Windspeed VS Electricity Consumption")
print('Coefficients迴歸係數: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)
lin_regplot(X3, y3, slr)
plt.xlabel('Windspeed [km/h]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()


# In[14]:


from sklearn.cross_validation import train_test_split
data = data[['temperature','pressure', 'windspeed','electricity_consumption']]
X = data.iloc[:, :].values
y = data['temperature'].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker='o',label='Training data')
plt.scatter(y_test_pred, y_test_pred-y_test, c='lightgreen', marker='s',label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')

plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')

plt.xlim([-10, 50])
plt.tight_layout()
plt.show()


# In[15]:


#MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

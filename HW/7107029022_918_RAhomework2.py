
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


# 印出data的資料類型
print(type(data))


# In[4]:


print("資料鍵值：",data.keys(),"\n")
print("資料大小：",data.shape,"\n")
print("溫度資料：","\n",data.temperature,"\n")
print("氣壓資料：","\n",data.pressure,"\n")
print("風速資料：","\n",data.windspeed,"\n")
print("用電量資料：","\n",data.electricity_consumption)


# In[5]:


s = data[['temperature','pressure', 'windspeed','electricity_consumption']]
s.describe()


# In[6]:


# 搜尋特定的欄位，列出所有溫度、氣壓、風速、用電量資料大小
print("溫度資料大小：",data.temperature.shape)
print("氣壓資料大小：",data.pressure.shape)
print("風速資料大小：",data.windspeed.shape)
print("用電量資料大小：",data.electricity_consumption.shape)


# In[7]:


#印出前5個數據
data.head()


# In[8]:


# 溫度、氣壓、風速分別和用電量關係之散佈圖
data['temperature'] = data['temperature'].apply(lambda x:0.453592*x)
data.plot.scatter(x='temperature', y='electricity_consumption',s=1,c='b')
plt.title('Temperature VS Electricity Consumption')
plt.xlabel('Temperature')
plt.ylabel('Electricity Consumption')
plt.grid(True)

data['pressure'] = data['pressure'].apply(lambda x:0.453592*x)
data.plot.scatter(x='pressure', y='electricity_consumption',s=1,c='c')
plt.title('Pressure VS Electricity Consumption')
plt.xlabel('Pressure')
plt.ylabel('Electricity Consumption')
plt.grid(True)

data['windspeed'] = data['windspeed'].apply(lambda x:0.453592*x)
data.plot.scatter(x='windspeed', y='electricity_consumption',s=1,c='g')
plt.title('Windspeed VS Electricity Consumption')
plt.xlabel('Windspeed')
plt.ylabel('Electricity Consumption')
plt.grid(True)


# In[9]:


# 針對用電量設定類別
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
data


# In[10]:


# 看出分類
sns.FacetGrid(data, hue="electricity_consumption_category", size=8).map(plt.scatter, "temperature", "pressure",s=8).add_legend()
plt.title("Temperature & Pressure")
plt.xlabel("Temperature",size=14)
plt.ylabel("Pressure",size=14)
plt.show()

sns.FacetGrid(data, hue="electricity_consumption_category", size=8).map(plt.scatter, "pressure", "windspeed",s=8).add_legend()
plt.title("Pressure & Windspeed")
plt.xlabel("Pressure",size=14)
plt.ylabel("Windspeed",size=14)
plt.show()

sns.FacetGrid(data, hue="electricity_consumption_category", size=8).map(plt.scatter, "temperature", "windspeed",s=8).add_legend()
plt.title("Temperature & Windspeed")
plt.xlabel("Temperature",size=14)
plt.ylabel("Windspeed",size=14)
plt.show()


# In[11]:


from sklearn.linear_model import LinearRegression

X1 = data[['temperature']].values
y1 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X1, y1)
print("Temperature VS Electricity Consumption")
print('Slope斜率: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)
print('\n')
X2 = data[['pressure']].values
y2 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X2, y2)
print("Pressure VS Electricity Consumption")
print('Slope斜率: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)
print('\n')
X3 = data[['windspeed']].values
y3 = data['electricity_consumption'].values
slr = LinearRegression()
slr.fit(X3, y3)
print("Windspeed VS Electricity Consumption")
print('Coefficients迴歸係數: %.3f' % slr.coef_[0])
print('Intercept截距: %.3f' % slr.intercept_)


# In[12]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='c',s=1)
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return 

lin_regplot(X1, y1, slr)
plt.xlabel('Temperature [℃]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()

lin_regplot(X2, y2, slr)
plt.xlabel('Pressure [atm]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()

lin_regplot(X3, y3, slr)
plt.xlabel('Windspeed [km/h]')
plt.ylabel('Electricity Consumption [kWh]')
plt.show()


# In[13]:


# 用電量小於200
filter1 = data["electricity_consumption"] <= 200
a = data[filter1].shape[0]
print("用電量小於200kWh：",a)

# 用電量200~400
filter2 = data["electricity_consumption"] <= 400
b = data[filter2].shape[0]-data[filter1].shape[0]
print("用電量200kWh~400kWh：",b)

# 用電量400~600
filter3 = data["electricity_consumption"] <= 600
c = data[filter3].shape[0]-data[filter2].shape[0]
print("用電量400kWh~600kWh：",c)

# 用電量600~800
filter4 = data["electricity_consumption"] <= 800
d = data[filter4].shape[0]-data[filter3].shape[0]
print("用電量600kWh~800kWh：",d)

# 用電量800~1000
filter5 = data["electricity_consumption"] <= 1000
e = data[filter5].shape[0]-data[filter4].shape[0]
print("用電量800kWh~1000kWh：",e)

# 用電量1000~1200
filter6 = data["electricity_consumption"] <= 1200
f = data[filter6].shape[0]-data[filter5].shape[0]
print("用電量1000kWh~1200kWh：",f)

# 用電量大於1200
filter7 = data["electricity_consumption"] > 1200
g = data[filter7].shape[0]
print("用電量大於1200kWh：",g)


# In[14]:


# 上述統計數據建立新表格
ElectricityConsumption = ['<200kWh', '200kWh~400kWh', '400kWh~600kWh', '600kWh~800kWh', '800kWh~1000kWh', '1000kWh~1200kWh', '>1200kWh']
Counts = [a,b,c,d,e,f,g]
df = pd.DataFrame()
df["Electricity Consumption"] = ElectricityConsumption
df["Counts"] = Counts
df


# In[15]:


# 指定主題的風格參數
sns.set(style="whitegrid",context='notebook')

# 設置圖表大小
f, ax = plt.subplots(figsize=(15, 5))

# 繪製長條圖(barplot)
sns.set_color_codes("pastel")
# 設定x軸變量為Counts,y軸變量為Electricity Consumption。
sns.barplot(x="Counts", y="Electricity Consumption", data=df, label="Total", color="b")

# 新增圖例和座標軸，frameon:是否繪製圖像邊緣
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Electricity Consumption", xlabel="Counts")
sns.despine(left=True, bottom=True)

ax.tick_params(axis='x',labelsize=12) # x轴
ax.tick_params(axis='y',labelsize=12) # y轴

# 顯示圖表
plt.tight_layout()
plt.show()


# In[16]:


data = data[['temperature','pressure', 'windspeed','electricity_consumption']]

# 分析特徵的兩兩相關
colormap = plt.cm.viridis
plt.figure(figsize=(7,7))
plt.title('Electricity Consumption of Features', y=1.05, size=16)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


# In[17]:


# 分析特徵的兩兩相關
sns.pairplot(data)
plt.show()


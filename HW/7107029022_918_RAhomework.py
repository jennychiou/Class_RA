
# coding: utf-8

# In[1]:


#匯入模組
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[2]:


# 讀取數據
data = pd.read_csv('data/500_Person_Gender_Height_Weight_Index.csv')


# In[3]:


# 印出data的資料類型
print(type(data))


# In[4]:


print("資料鍵值：",data.keys())
print("資料大小：",data.shape)
print("性別資料：",data.Gender)
print("身高資料：",data.Height)
print("體重資料：",data.Weight)


# In[5]:


data.describe()


# In[6]:


# 搜尋特定的欄位，列出所有性別、身高、體重欄位(不重複)
Height_Cols = data.drop_duplicates(subset=['Height'],keep='first')
print(Height_Cols['Height'])
print("Height資料大小：",Height_Cols.shape)

Weight_Cols = data.drop_duplicates(subset=['Weight'],keep='first')
print(Weight_Cols['Weight'])
print("Weight資料大小：",Weight_Cols.shape)


# In[7]:


#印出前5個數據
data.head()


# In[8]:


# 更改Index欄位名稱
data.rename(columns={'Index': 'Type'}, inplace=True)
data.head()


# In[9]:


# 針對Type欄位原始數據進行修改(0-Extremely Weak,1-Weak,2-Normal,3-Overweight,4-Obesity,5-Extreme Obesity)
d = {0:'Extremely Weak',1:'Weak',2:'Normal',3:'Overweight',4:'Obesity',5:'Extreme Obesity'}
data = data.replace(d)
print(data)


# In[10]:


data['Weight'] = data['Weight'].apply(lambda x:0.453592*x)
data.plot.scatter(x='Weight', y='Height')
plt.show()


# In[11]:


# 身高
x = data['Height']
# 體重
y = data['Weight']
plt.title('Data of Height and Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid(True)
plt.scatter(x,y)
plt.show()


# In[12]:


# 看出分類(0-Extremely Weak,1-Weak,2-Normal,3-Overweight,4-Obesity,5-Extreme Obesity)
sns.FacetGrid(data, hue="Gender", size=5).map(plt.scatter, "Height", "Weight").add_legend()
sns.FacetGrid(data, hue="Type", size=5).map(plt.scatter, "Height", "Weight").add_legend()
plt.show()


# In[13]:


X = data[['Weight']].values
y = data['Height'].values

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# In[14]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return 

lin_regplot(X, y, slr)
plt.xlabel('Weight [KG]')
plt.ylabel('Height [CM]')
plt.tight_layout()
plt.show()


# In[15]:


# 挑選欄位名稱為Type(會有重複)
s = data[['Type']]

# 根據Index每一項進行統計，統計之結果放置在新欄位counts，並以遞減排序方式呈現。
index_count = pd.DataFrame(s.groupby('Type').size().sort_values(ascending=True).rename('counts').reset_index())
index_count.head(6)


# In[16]:


# 指定主題的風格參數
sns.set(style="whitegrid",context='notebook')

# 設置圖表大小
f, ax = plt.subplots(figsize=(10, 5))


# 繪製長條圖(barplot)
sns.set_color_codes("pastel")
# 設定x軸變量為counts,y軸變量為Primary Type。
sns.barplot(x="counts", y="Type", data=index_count, label="Total", color="b")

# 新增圖例和座標軸
# frameon:是否繪製圖像邊緣
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Type", xlabel="Counts")
sns.despine(left=True, bottom=True)

# 顯示圖表
plt.tight_layout()
plt.show()


# In[17]:


# 將資料數值化
data = pd.read_csv('data/500_Person_Gender_Height_Weight_Index.csv')
# 更改欄位名稱
data.rename(columns={'Index': 'Type'}, inplace=True)
# 針對Index欄位原始數據進行修改(0 - Male,1 - Female)
d = {'Male':0,'Female':1}
data = data.replace(d)

# 分析特徵的兩兩相關
colormap = plt.cm.viridis
plt.figure(figsize=(7,7))
plt.title('BMI of Features', y=1.05, size=16)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


# In[18]:


# 分析特徵的兩兩相關
sns.pairplot(data)
plt.show()


# In[19]:


# 依據性別統計各種Type的個數(0-Extremely Weak,1-Weak,2-Normal,3-Overweight,4-Obesity,5-Extreme Obesity)
g = sns.FacetGrid(data, col='Gender')
g.map(plt.hist,'Type', bins=6)
plt.show()


# In[20]:


# hue參數的把兩個直方圖畫在同一張圖上，其中有size參數决定高度，aspect長寬比，width=size*aspect
grid = sns.FacetGrid(data, hue='Gender')
grid.map(plt.hist,'Type', alpha=0.5, bins=6)
grid.add_legend()
plt.show()


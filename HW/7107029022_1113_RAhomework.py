
# coding: utf-8

# # Naive Bayes 

# In[1]:


#匯入模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


# In[2]:


#匯入資料集
data = pd.read_csv('data/Titanic Disaster dataset.csv')


# In[3]:


# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                  np.where(data["Embarked"]=="C",1,np.where(data["Embarked"]=="Q",2,3)))


# In[4]:


data=data[["Survived","Pclass","Sex_cleaned","Age","SibSp","Parch","Fare","Embarked_cleaned"]].dropna(axis=0, how='any')


# In[5]:


data.head()


# In[6]:


X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))


# In[7]:


gnb = GaussianNB()
used_features =["Pclass","Sex_cleaned","Age","SibSp","Parch","Fare","Embarked_cleaned"]


# In[8]:


gnb.fit(X_train[used_features].values,X_train["Survived"])
y_pred = gnb.predict(X_test[used_features])


# In[9]:


print("Number of mislabeled points out of a total {} points : {} \nPerformance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])))


# In[10]:


mean_survival=np.mean(X_train["Survived"])
mean_not_survival=1-mean_survival
print("Survival prob = {:03.2f}%, Not survival prob = {:03.2f}%".format(100*mean_survival,100*mean_not_survival))


# In[11]:


mean_fare_survived = np.mean(X_train[X_train["Survived"]==1]["Fare"])
std_fare_survived = np.std(X_train[X_train["Survived"]==1]["Fare"])
mean_fare_not_survived = np.mean(X_train[X_train["Survived"]==0]["Fare"])
std_fare_not_survived = np.std(X_train[X_train["Survived"]==0]["Fare"])

print("mean_fare_survived = {:03.2f}".format(mean_fare_survived))
print("std_fare_survived = {:03.2f}".format(std_fare_survived))
print("mean_fare_not_survived = {:03.2f}".format(mean_fare_not_survived))
print("std_fare_not_survived = {:03.2f}".format(std_fare_not_survived))


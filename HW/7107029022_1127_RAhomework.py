
# coding: utf-8

# <img src="img/1127-1.jpg" width="30%">

# In[1]:


states = ["Sunny","Cloudy","Rainy"]
transitionName = [["SS","SC","SR"],["CS","CC","CR"],["RS","RC","RR"]]
matrix = [0.63,0.17,0.2]
weather = [[0.5,0.25,0.25],[0.375,0.125,0.375],[0.125,0.625,0.375]]
array = [[0.6,0.25,0.05],[0.2,0.25,0.1],[0.15,0.25,0.35],[0.05,0.25,0.5]]


# # weather matrix

# <img src="img/1127-2.jpg" width="70%">

# # transition matrix

# <img src="img/1127-3.jpg" width="70%">

# In[2]:


#a1
a1 = []
a1s = matrix[0] * array[0][0]
a1c = matrix[1] * array[0][1]
a1r = matrix[2] * array[0][2]
a1.append(a1s)
a1.append(a1c)
a1.append(a1r)

print("a1s :",a1s)
print("a1c :",a1c)
print("a1r :",a1r)


# In[3]:


a1


# In[4]:


#a2
a2 = []
a2s = (a1[0]*weather[0][0]+a1[1]*weather[0][1]+a1[2]*weather[0][2]) * array[1][0]
a2c = (a1[0]*weather[1][0]+a1[1]*weather[1][1]+a1[2]*weather[1][2]) * array[1][1]
a2r = (a1[0]*weather[2][0]+a1[1]*weather[2][1]+a1[2]*weather[2][2]) * array[1][2]
a2.append(a2s)
a2.append(a2c)
a2.append(a2r)

print("a2s :",a2s)
print("a2c :",a2c)
print("a2r :",a2r)


# In[5]:


a2


# In[6]:


#a3
a3 = []
a3s = (a2[0]*weather[0][0]+a2[1]*weather[0][1]+a2[2]*weather[0][2]) * array[2][0]
a3c = (a2[0]*weather[1][0]+a2[1]*weather[1][1]+a2[2]*weather[1][2]) * array[2][1]
a3r = (a2[0]*weather[2][0]+a2[1]*weather[2][1]+a2[2]*weather[2][2]) * array[2][2]
a3.append(a3s)
a3.append(a3c)
a3.append(a3r)

print("a3s :",a3s)
print("a3c :",a3c)
print("a3r :",a3r)


# In[7]:


a3


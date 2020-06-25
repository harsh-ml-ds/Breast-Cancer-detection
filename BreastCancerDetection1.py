#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Creation of arrays
#from list
arr1=np.array([[1,2,4],[5,8,7]], dtype='float')
print("arr1:{}".format(arr1))

#from tuple
arr2=np.array((1,3,2))
print("arr2:{}".format(arr2))

#from 2X3 with zeros
arr3=np.array((2,3))
print('arr3:{}'.format(arr3))

#in range from 0 to 20 with step 2
arr4=np.arange(0,20,2)
print('arr4: {}'.format(arr4))


# In[3]:


#in range 20 values from 0 to 10
arr5=np.linspace(0,10,20)
print("arr5: {}".format(arr5))


# In[4]:


##making an array of consecutive natural numbers
arr6=np.arange(15).reshape(3,5)
print("arr6: {}".format(arr6))


# In[5]:


import pandas as pd


# In[6]:


df=pd.read_excel('Week 1 Dataset.xlsx')


# In[7]:


df.head()


# In[8]:


df.columns


# In[15]:


df.info()


# In[16]:


df1=df.rename(columns={'State':'State_name'})


# In[17]:


df.rename(columns={'State':'State_name'},inplace=True)


# In[18]:


df[['Age', 'Name', 'State_name']]


# In[19]:


df.iloc[0:3, 0:2]


# In[20]:


df.isna().sum()


# In[21]:


df.describe()


# In[22]:


df.Sex.value_counts()


# In[23]:


import seaborn as sns


# In[25]:


sns.pairplot(df, hue='Number of siblings')


# In[26]:


df.corr(method='pearson')


# In[27]:


sns.heatmap(df.corr(), annot=True, fmt='0.0%')


# In[29]:


#Part-2


# In[ ]:





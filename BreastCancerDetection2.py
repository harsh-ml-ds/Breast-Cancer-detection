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


# In[9]:


df.info()


# In[10]:


df1=df.rename(columns={'State':'State_name'})


# In[11]:


df.rename(columns={'State':'State_name'},inplace=True)


# In[12]:


df[['Age', 'Name', 'State_name']]


# In[13]:


df.iloc[0:3, 0:2]


# In[14]:


df.isna().sum()


# In[15]:


df.describe()


# In[16]:


df.Sex.value_counts()


# In[17]:


import seaborn as sns


# In[18]:


sns.pairplot(df, hue='Number of siblings')


# In[19]:


df.corr(method='pearson')


# In[20]:


sns.heatmap(df.corr(), annot=True, fmt='0.0%')


# In[21]:


#Part-2


# In[22]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df.iloc[:,1]=label_encoder.fit_transform(df.iloc[:,1].values)


# In[23]:


df.iloc[:,1]


# In[24]:


#Part-3


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


df_n=pd.read_csv('cancer dataset.csv')


# In[27]:


df_n.head()


# In[28]:


df_n.columns


# In[29]:


x=df_n.iloc[:,2:31].values   #features that help us determine if patient has cancer or not
y=df_n.iloc[:,1].values     #this is the dataset containing our target variable which indicates diagnosis


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.25, random_state=0)


# In[34]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[35]:


def logreg(x_train, y_test):
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(x_train, y_train)
    print("Logistic Regression training Accurracy", log.score(x_train, y_train))
    return log


# In[37]:


logrex=logreg(x_train, y_train)


# In[38]:


logrex


# In[39]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, logrex.predict(x_test))
print(cm)


# In[41]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print("Testing accuracy of Logistic Regression: ", ((TP+TN)/(TP+TN+FN+FP)))


# In[ ]:





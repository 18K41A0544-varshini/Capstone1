#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import math


# In[66]:


brstcancerdata = pd.read_csv('data.csv')
brstcancerdata.head(10)


# In[67]:


print("no of records: ", len(brstcancerdata))
#brstcancerdata.drop('Unnamed: 0', axis=1)


# # Analyzing data
# 

# In[68]:


sb.countplot(x="diagnosis",data=brstcancerdata)


# In[69]:


brstcancerdata.info()


# # data wraggling

# In[70]:


brstcancerdata.isnull()


# In[71]:


brstcancerdata.isnull().sum()


# In[72]:


brstcancerdata.head()


# In[73]:


brstcancerdata.drop(['Unnamed: 32','id'],axis=1,inplace=True)
brstcancerdata.dropna()


# In[74]:


brstcancerdata.head()


# In[75]:


brstcancerdata.isnull().sum()


# In[76]:


d=pd.get_dummies(brstcancerdata["diagnosis"],drop_first=True)
d.head()


# In[77]:


brstcancerdata=pd.concat([brstcancerdata,d],axis=1)
brstcancerdata.head()


# In[78]:


brstcancerdata.drop(['diagnosis'],axis=1,inplace=True)
brstcancerdata.head()


# # Train Data

# In[79]:


#Independent variable
X= brstcancerdata.drop(['M'],axis=1)
#Dependent variable
y=brstcancerdata['M']


# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[82]:


from sklearn.linear_model import LogisticRegression


# In[189]:


logmodel=LogisticRegression(max_iter=10000)


# In[190]:


logmodel.fit(X_train,y_train)


# In[191]:


predictions=logmodel.predict(X_test)


# In[192]:


from sklearn.metrics import classification_report


# In[193]:


classification_report(y_test,predictions)


# In[194]:


from sklearn.metrics import confusion_matrix


# In[195]:


confusion_matrix(y_test,predictions)


# In[196]:


from sklearn.metrics import accuracy_score


# In[197]:


accuracy_score(y_test,predictions)


# # input

# In[198]:


input=[17.99,1.38,522.80,1001.0,0.11840,0.27760,0.3001,0.14710,0.2419,0.07871,1.0950,1.9053,8.589,153.40,0.006399,
    0.04904,0.05373,0.11587,0.03003,0.006193,26.38,17.33,184.60,2019.0,0.1622,0.6656,0.7119,0.9654,0.4601,0.011890]
input_array=np.asarray(input)
print(input)
input_reshaped=input_array.reshape(1,-1)
predict=logmodel.predict(input_reshaped)
print(predict)
if(predict[0]==1):
    print("MALIGNANT_____MAY RECCUR")
else:
    print("BENIGN_______LESS CHANCES TO RECCUR")


# # SVM

# # Test

# In[199]:


from sklearn.svm import SVC


# In[201]:


logmodel1=SVC(max_iter=10000)


# In[202]:


logmodel1.fit(X_train,y_train)


# # Predictions

# In[203]:


p1=logmodel1.predict(X_test)


# In[204]:


print(confusion_matrix(y_test,p1))


# In[205]:


print(classification_report(y_test,p1))


# In[206]:


accuracy_score(y_test,p1)


# In[207]:


#here we can improve the accuracy ny two ways:

#1. Normalization

#2. HyperParameter tuning


# # Improving model performance

# In[208]:


from sklearn import preprocessing


# In[209]:


min_max_s=preprocessing.MinMaxScaler()


# In[210]:


X_train_scaled=min_max_s.fit_transform(X_train)
X_test_scaled=min_max_s.fit_transform(X_test)


# In[211]:


logmodel1.fit(X_train_scaled,y_train)


# In[212]:


y_predict=logmodel1.predict(X_test_scaled)
print(y_predict)


# In[213]:


cm=confusion_matrix(y_test,y_predict)
print(cm)


# In[214]:


print(classification_report(y_test,y_predict))


# In[215]:


accuracy_score(y_test,y_predict)


# # input

# In[216]:


input=[17.99,1.38,522.80,1001.0,0.11840,0.27760,0.3001,0.14710,0.2419,0.07871,1.0950,1.9053,8.589,153.40,0.006399,
    0.04904,0.05373,0.11587,0.03003,0.006193,26.38,17.33,184.60,2019.0,0.1622,0.6656,0.7119,0.9654,0.4601,0.011890]
input_array=np.asarray(input)
print(input)
input_reshaped=input_array.reshape(1,-1)
predict1=logmodel1.predict(input_reshaped)
print(predict1)
if(predict1[0]==1):
    print("MALIGNANT_____MAY RECCUR")
else:
    print("BENIGN_______LESS CHANCES TO RECCUR")


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[9]:


data_fake = pd.read_csv(r"C:\Users\Ananya Kaul\OneDrive\Desktop\Fake.csv")
data_true = pd.read_csv(r"C:\Users\Ananya Kaul\OneDrive\Desktop\True.csv")


# In[10]:


data_fake.head()


# In[11]:


data_true.head()


# In[12]:


data_fake.tail()


# In[13]:


data_true.tail()


# In[14]:


data_fake["class"]=0
data_true["class"]=1


# In[15]:


data_fake.shape, data_true.shape


# In[17]:


data_fake_manual_testing =data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis=0, inplace = True)
    
data_true_manual_testing =data_true.tail(10)
for i in range(21416 ,21406,-1):
    data_true.drop([i], axis =0, inplace = True)


# In[18]:


data_fake.shape, data_true.shape


# In[108]:


data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1


# In[109]:


data_fake_manual_testing.head(10)


# In[110]:


data_true_manual_testing.head(10)


# In[111]:


data_merge = pd.concat([data_fake, data_true], axis=0 )
data_merge.head(10)


# In[112]:


data_merge.columns


# In[113]:


data = data_merge.drop(['title','subject','date'], axis =1)


# In[114]:


data.isnull().sum()


# In[115]:


data = data.sample(frac=1)


# In[116]:


data.head()


# In[117]:


data.reset_index(inplace= True)
data.drop(['index'], axis=1, inplace=True)


# In[118]:


data.columns


# In[119]:


data.head()


# In[120]:


def wordopt(text):
    text= text.lower()
    text= re.sub('\[.*?\]', '', text)
    text= re.sub("\\W", " ", text)
    text= re.sub('https?://\S+|www\.\S+','',text)
    text= re.sub('<.*?>+','', text)
    text= re.sub('[%s]'% re.escape(string.punctuation),'', text)
    text= re.sub('\n', '', text)
    text= re.sub('\w*\d\w*','', text)
    return text


# In[121]:


data['text']= data['text'].apply(wordopt)


# In[122]:


x= data['text']
y= data['class']


# In[123]:


x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.25)


# In[124]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train= vectorization.fit_transform(x_train)
xv_test= vectorization.transform(x_test)


# In[125]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[126]:


pred_lr = LR.predict(xv_test)


# In[127]:


LR.score(xv_test, y_test)


# In[128]:


print(classification_report(y_test, pred_lr))


# In[129]:


from sklearn.tree import DecisionTreeClassifier
DT= DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[130]:


pred_dt= DT.predict(xv_test)


# In[134]:


DT.score(xv_test, y_test)


# In[135]:


print(classification_report(y_test, pred_dt))


# In[139]:


from sklearn.ensemble import GradientBoostingClassifier
GB= GradientBoostingClassifier(random_state= 0)
GB.fit(xv_train, y_train)


# In[140]:


pred_gb=GB.predict(xv_test)


# In[141]:


GB.score(xv_test, y_test)


# In[142]:


print(classification_report(y_test, pred_gb))


# In[143]:


from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(random_state= 0)
RF.fit(xv_train, y_train)


# In[144]:


pred_rf=RF.predict(xv_test)


# In[145]:


RF.score(xv_test, y_test)


# In[146]:


print(classification_report(y_test, pred_rf))


# In[156]:


def output_label(n):
    if n== 0:
        return "Fake News"
    elif n ==1:
        return "Not a Fake News"

def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test= new_def_test["text"]
    new_xv_test= vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    pred_GB=GB.predict(new_xv_test)
    pred_RF=RF.predict(new_xv_test)
    
    return print("\n\nLR Prediction: {} \nDT Prediction:  {} \nGB Prediction: {} \nRF Prediction: {}".format(output_label(pred_LR[0]),output_label(pred_DT[0]),output_label(pred_GB[0]),output_label(pred_RF[0])))


# In[167]:


news= str(input())
manual_testing(news)


# In[163]:


news= str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





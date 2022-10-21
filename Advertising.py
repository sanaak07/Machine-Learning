#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


# In[5]:


data=pd.read_csv('advertising.csv', index_col=0)
data.head()
data.columns=['TV','Radio','Newspaper','Sales']


# In[6]:


data.shape


# In[8]:


fig,axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


# In[11]:


feature_cols=['TV'] 
x=data[feature_cols]   #independent variable
y=data.Sales              #dependent variable


# In[12]:


from sklearn.linear_model import LinearRegression   #importing linear regression model from sklearn
lm=LinearRegression()                                             # initialising the model
lm.fit(x,y)                                                               #fitting the model on x and y


# In[13]:


print(lm.intercept_)
print(lm.coef_)             #by interpreting the model co-eff, we can say that a unit increase in tv ad spending is associated by 0.04753664 increase in sales


# In[14]:


7.032594+0.047537*50   #lets predict the sales for spending $50,000 on advertisement


# In[17]:


X_new=pd.DataFrame({'TV':[50]})    #lets predict new x values for sales in the market
X_new.head()  


# In[18]:


lm.predict(X_new)    #predict sales of tv widgets


# In[20]:


X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()


# In[21]:


preds=lm.predict(X_new)
preds


# In[24]:


data.plot(kind='scatter',x='TV',y='Sales')
plt.plot (X_new,preds,c='red',linewidth=2)


# In[26]:


import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales~TV',data=data).fit()


# In[27]:


lm.conf_int()


# In[29]:


lm.pvalues   #here the difference is very less between the values (<0.05), hence we confer that there IS a relation between the ads and tv 


# In[30]:


lm.rsquared    #rsquared value is most useful while comparing different models


# In[33]:


feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

from sklearn import model_selection
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.3, random_state=42)


# In[35]:


lm=LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)


# In[36]:


lm=LinearRegression()
lm.fit(xtrain,ytrain)


# In[37]:


print(lm.intercept_)
print(lm.coef_)

predictions=lm.predict(xtest)
print(sqrt(mean_squared_error(ytest,predictions)))


# In[39]:


lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()


# In[40]:


lm=smf.ols(formula='Sales~TV+Radio',data=data).fit()
lm.rsquared


# In[42]:


lm=smf.ols(formula='Sales~TV+Radio+Newspaper',data=data).fit()
lm.rsquared


# In[43]:


import numpy as np
np.random.seed(12345)

nums=np.random.rand(len(data))
mask_large=nums>0.5

data['Size']='small'
data.loc[mask_large,'Size']='large'
data.head()


# In[45]:


data['IsLarge']=data.Size.map({'small':0,'large':1})
data.head()


# In[46]:


feature_cols=['TV','Radio','Newspaper','IsLarge']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

zip(feature_cols,lm.coef_)


# In[47]:


np.random.seed(123456)

nums=np.random.rand(len(data))
mask_suburban=(nums>0.33) & (nums <0.66)
mask_urban=nums>0.66
data['Area']='rural'
data.loc[mask_suburban,'Area']='suburban'
data.loc[mask_urban,'Area']='urban'
data.head()


# In[49]:


area_dummies=pd.get_dummies(data.Area,prefix='Area').iloc[:,1:]

data=pd.concat([data,area_dummies],axis=1)
data.head()


# In[50]:


feature_cols=['TV','Radio','Newspaper','IsLarge','Area_suburban','Area_urban']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)
print(feature_cols,lm.coef_)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Collection made by a movie using Decision tree, ensemble learning with boasting

# So in this exercise we are going to use a data set to predict the money collection made by a movie based 17 parameters. So out data set has 18 columns and 506 data points. Most of them are numerical data but there are 2 catagorical variables.Lets dive into the problem by first importing the required packages.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('Movie_data.csv',header=0)
data.head()


# In[3]:


data.info()


# We find that only Time_taken feature has some missing values and the rest of the columns have all the required data. Since this is simulated data and not real time data, most of the data is present. In case of real time data, many features would have missing data. Let now impute our missing data with the median of the Time_Taken feature data.

# In[4]:


data.isnull().sum()


# We now find that 12 data points are missing in the Time_taken feature. Now lets imput the data

# In[5]:


median=data['Time_taken'].median()
#print(median)
data['Time_taken'].fillna(value=median,inplace=True)
#print(data.isnull().sum().sum())
data.info()


# Now we convert the catagorical variable data into numerical values by using the get_dummies method from pandas

# In[6]:


data=pd.get_dummies(data,columns=['3D_available','Genre'],drop_first= True)
data.head()


# Now we create out X and y variables.

# In[7]:


Y=data['Collection']
X=data.loc[:, data.columns!='Collection']
a=["Marketing expense","Production expense","Multiplex coverage","Budget","Movie_length","Lead_ Actor_Rating","Lead_Actress_rating","Director_rating","Producer_rating","Critic_rating","Trailer_views","Time_taken","Twitter_hastags","Avg_age_actors","Num_multiplex","3D_available_YES","Genre_Comedy","Genre_Drama","Genre_Thriller"]


# Now since the features are not in the scale, we scale the X data so that all the features have the same impact on the output

# In[8]:


from sklearn.preprocessing import StandardScaler 
scalar=StandardScaler()
X_Scaled=scalar.fit_transform(X)
X_Scaled


# Now we split the data into testing and training data using the test_train_split fucntion from sklearn

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_Scaled,Y,test_size=0.2, random_state=31)


# ## Controlling the growth of a tree
# Now we use the Decision tree Regression model to train our model.
# In any decision tree model, the model can be controlled from overifiitng and excess computation by pruning the tree, basically restricting the growth of the tree with respect to splitting.
# There are 3 ways of prunning:
# 1. Maximum levels in the tree: This is basically restricting the number of splits or layers in the tree.
# 2. Minimum number of observations in the at the internal node: In the hyperparameter, the node splits only if the number of samples present at that node is more than the specified minimum.
# 3.  Minimum number of observations at the leaf node: In this hyperparameter we can control the number of samples in the leaf node, the leaf node will have more the specified number of samples.

# In[10]:


from sklearn.tree import DecisionTreeRegressor
DectreeReg=DecisionTreeRegressor(max_depth=5)
DectreeReg.fit(X_train,y_train)


# Now we find the accuracy of the model

# In[11]:


DectreeReg.score(X_test,y_test)


# Now we plot the decision tree

# In[12]:


from sklearn import tree
dot_data=tree.export_graphviz(DectreeReg,out_file=None,feature_names=a)
from IPython.display import Image
import pydotplus


# In[13]:


graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# We can see that the growth of the tree can be restricted by prunning and thus preventing overfitting.

# ## Ensemble Learning
# Since decision trees have a lot of variance, we can use emsemble learning to improve our model. There are 3 types of Ensemble learning:
# 1. Bagging
# 2. Random Forests
# 3. Boasting
# 

# ### Bagging
# Let's first start with Bagging technique

# In[14]:


from sklearn import tree
classifier=tree.DecisionTreeRegressor()


# In[15]:


from sklearn.ensemble import BaggingRegressor


# In[16]:


Dec_bag=BaggingRegressor(base_estimator=classifier,n_estimators=1000,bootstrap=True,random_state=31)
Dec_bag.fit(X_train,y_train)


# In[17]:


Dec_bag.score(X_test,y_test)


# With bagging we can see a clear increase in the accuracy

# ## Random Forest
# Lets try with random forest 
# 

# In[18]:


from sklearn.ensemble import RandomForestRegressor


# In[19]:


rf=RandomForestRegressor(n_estimators=1000,random_state=31)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)


# This is again better than our original Decision tree

# ## Boasting
# Lets try boasting techniques
# 

# The major difference between boasting and bagging is that the growth of the tree is not controlled.
# There are 3 types of boasting:
# 1. Gradient Boost
# 2. Ada boost
# 3. XG boost

# ### Gradient boost
# Lets start with Gradient Boast

# In[20]:


from sklearn.ensemble import GradientBoostingRegressor


# In[21]:


gbr=GradientBoostingRegressor(learning_rate=0.02,n_estimators=1000)
gbr.fit(X_train,y_train)
gbr.score(X_test,y_test)


# We clearly get a better accuracy in booting. This is because booting focusses on the results with largest residuals.

# ### Ada boost
# Now we try to compute the model with ada boost technique

# In[22]:


from sklearn.ensemble import AdaBoostRegressor


# In[23]:


abr=AdaBoostRegressor(base_estimator=rf,learning_rate=0.02,n_estimators=100)
abr.fit(X_train,y_train)
abr.score(X_test,y_test)


# In[ ]:





# ### XG boost

# Lets now try with XG boost

# In[24]:


import xgboost as xgb


# In[26]:


xbr=xgb.XGBRegressor(max_depth=5,n_estimators=10000,learning_rate=0.2,n_jobs=-1,gamma=0.1)
xbr.fit(X_train,y_train)
xbr.score(X_test,y_test)


# In[29]:


xgb.plot_importance(xbr)


# #### XGBoost gives the best accuracy of 84.85%

# We can clearly see the difference in the accuracy score between a regular decision tree model and a boosted model. There is a significant difference and Boosting techinques give the best results.
# 

# Now we can use grid search to tune our model to get the best results

# In[35]:


xbr=xgb.XGBRegressor(learning_rate=0.2,n_estimators=250,random_state=31)
param={
    'max_depth':range(3,9,3),
    'gamma':[0.1,0.2,0.3],
    'subsample':[0.8,0.9],
    'colsample_bytree':[0.8,0.9],
    'reg_alpha':[0.001,0.1,1]
}


# In[40]:


from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(xbr,param,n_jobs=-1,cv=5)


# In[41]:


grid_search.fit(X_train,y_train)


# In[49]:


grid_search.best_params_


# In[50]:


xgb_best=xgb.XGBRegressor(colsample_bytree=0.8,
 gamma= 0.2,
 max_depth=6,
 reg_alpha= 0.001,
 subsample=0.9)


# In[51]:


xgb_best.fit(X_train,y_train)
xgb_best.score(X_test,y_test)


# In[ ]:





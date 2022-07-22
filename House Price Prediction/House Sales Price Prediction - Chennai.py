#!/usr/bin/env python
# coding: utf-8

# #### Importing Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)


# #### Problem Statement: We have to predict the House sales price for Areas in Chennai from dataset provided
# * We have Predefined output to validate our model, So we do Supervised Learning.
# * The Predicted target is a continuous value, which implies us to do Regression Models

# #### Reading Dataset

# In[2]:


House_data=pd.read_csv("train-chennai-sale.csv")


# Analysing Basic Details of dataset

# In[3]:


House_data.shape


# Summary:Dataset Contains 7109 Rows and 22 Columns

# In[4]:


House_data.head()


# In[5]:


House_data.info()


# Summary: To count non-null rows and data types of all columns and we can see there are null values in some columns:
# N_BEDROOM: 1
# N_BATHROOM: 5
# QS_OVERALL: 48

# In[6]:


House_data.describe()


# Summary: Statistical terms of each valued column

# #### Working on null values

# In[7]:


House_data.isnull().sum()


# Summary: Null Values existing in columns N_BEDROOM:1,N_BATHROOM:5,QS_OVERALL:48.

# ##### Null Values: N_BEDROOM

# In[8]:


House_data[House_data['N_BEDROOM'].isnull()]


# Considering 50th and 75th percentile and mode of features INT_SQFT, N_ROOM, N_BATHROOM, we can fill the N_BEDROOM with 1

# In[9]:


House_data['N_BEDROOM'].fillna(value=1,inplace=True)


# In[10]:


House_data['N_BEDROOM'].isnull().sum()


# Summary:We have 0 null values in N_BEDROOM feature as we filled it with 1

# ##### Null Values: N_BATHROOM

# In[11]:


House_data[House_data['N_BATHROOM'].isnull()]


# Considering the 75th percentile of features INT_SQFT, N_ROOM, N_BEDROOM, we are filling null values in N_BATHROOM with 1

# In[12]:


House_data['N_BATHROOM'].fillna(value=1,inplace=True)


# In[13]:


House_data['N_BATHROOM'].isnull().sum()


# Summary:We have 0 null values in N_BATHROOM feature as we filled it with 1

# #### Null Values: QS_OVERALL

# In[14]:


House_data[House_data['QS_OVERALL'].isnull()]


# In[15]:


House_data['QS_OVERALL'].mean()


# In[16]:


House_data['QS_OVERALL'].fillna(value=3.50,inplace=True)


# In[17]:


House_data['QS_OVERALL'].isnull().sum()


# Summary: We have filled the null values in QS_OVERALL with the mean value of feature

# In[18]:


House_data.isnull().sum()


# ### Summary: Now we are free from Null Values

# #### Check for Duplicates

# In[19]:


House_data.duplicated().sum()


# Summary: We could see that there are no duplicates in the dataset

# #### Converting Data types

# In[20]:


House_data.dtypes


# we could see that DATE_SALE, N_BEDROOM, N_BATHROOM, DATE_BUILD data types should be changed

# In[21]:


House_data=House_data.astype({"N_BEDROOM":int,"N_BATHROOM":int})


# In[22]:


House_data['DATE_SALE']=pd.to_datetime(House_data['DATE_SALE'])


# In[23]:


House_data['DATE_BUILD']=pd.to_datetime(House_data['DATE_BUILD'])


# In[24]:


House_data[['DATE_SALE','DATE_BUILD','N_BEDROOM','N_BATHROOM']].dtypes


# Summary: we have successfully completed coverting features to right data types.

# ### Removing and adding customized features

# Remove PRT_ID from dataset as its unique values.

# In[25]:


House_data.drop('PRT_ID',axis=1,inplace=True)


# We can now create new feature "Age_Building" with years difference btw Date_Sale and Date_Build

# In[26]:


House_data["Age_Bldg"]=(House_data["DATE_SALE"]-House_data["DATE_BUILD"]).astype('timedelta64[Y]').astype('int')


# The total price of House can be included with Registration fees and also Commission 

# In[27]:


House_data['Total_Price'] = House_data['SALES_PRICE']+House_data['REG_FEE']+House_data['COMMIS']


# Now, we can remove the registration fees, commision and sales price as we created new combined feature Total_price

# In[28]:


House_data.drop(['REG_FEE','COMMIS','SALES_PRICE'],axis=1,inplace=True)


# In[29]:


House_data['YEAR_BUILD'] = House_data['DATE_BUILD'].dt.year
House_data['YEAR_SALE'] = House_data['DATE_SALE'].dt.year


# In[30]:


House_data.drop(['DATE_BUILD','DATE_SALE'],axis=1,inplace=True)


# # Exploratory Data Analysis
# ##### Lets explore each feature

# ### 1. Area

# In[31]:


House_data['AREA'].unique()


# Summary: we can see that there are same Areas with different spellings, we have to replace all incorrect names

# In[32]:


House_data['AREA']=House_data['AREA'].replace(['Karapakam','Ann Nagar','TNagar','Adyr','Velchery','Chormpet','KKNagar','Chrompt','Chrmpet','Chrompet','Ana Nagar'],['Karapakkam','Anna Nagar','T Nagar','Adyar','Velachery','Chromepet','KK Nagar','Chromepet','Chromepet','Chromepet','Anna Nagar'])


# In[33]:


House_data['AREA'].unique()


# #### Distribution plot

# In[34]:


House_data['AREA'].hist(bins=25)
plt.xlabel('AREA')
plt.ylabel("Count")
plt.show()


# Summary: Most buildings are from chromepet and karapakkam when compared with other areas

# #### Feature Vs Target plot

# In[35]:


House_data.groupby('AREA')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: we can clearly see the linear relation btw Area and Total_Price of the Building, so we can do label encoding starting with 1

# In[36]:


House_data["AREA"]=House_data["AREA"].replace({"Karapakkam":1,"Adyar":2,"Chromepet":3,"Velachery":4,"KK Nagar":5,"Anna Nagar":6,"T Nagar":7})


# ### 2. INT_SQFT

# #### Check outliers

# In[37]:


House_data['INT_SQFT'].describe()


# In[38]:


q3,q1=House_data.INT_SQFT.quantile(0.75),House_data.INT_SQFT.quantile(0.25)
iqr=q3-q1
iqr


# In[39]:


max_=q3+1.5*iqr
max_


# In[40]:


min_=q1-1.5*iqr
min_


# Summary: we dont have any outliers as all values exists in between max_ and min_

# #### Distribution plot

# In[41]:


House_data['INT_SQFT'].hist(bins=25)
plt.xlabel('INT_SQFT')
plt.ylabel("Count")
plt.show()


# Summary: Most no of buildings are in the range of 1500 and 1750 sqft

# #### Feature vs target

# In[42]:


plt.scatter(House_data['INT_SQFT'],House_data['Total_Price'])
plt.show()


# Summary: we can find a linear relation between INT_SQFT and Total_price of the building.

# ### 3. YEAR_SALE 

# #### Distribution Plot

# In[43]:


House_data['YEAR_SALE'].hist(bins=25)
plt.xlabel('YEAR_SALE')
plt.ylabel("Count")
plt.show()


# Summary: We can see uniform distribution with year of sale., Most sales happened btw 2008 to 2012

# #### Feature vs Target

# In[44]:


plt.scatter(House_data['YEAR_SALE'],House_data['Total_Price'])
plt.show()


# Summary: we cannot find a particular pattern btw YEAR_SALE and total_price, but there exists a relations, so we use this feature

# ### 4. DIST_MAINROAD

# #### Check for outliers

# In[45]:


House_data['DIST_MAINROAD'].describe()


# In[46]:


q3,q1=House_data.DIST_MAINROAD.quantile(0.75),House_data.DIST_MAINROAD.quantile(0.25)
iqr=q3-q1
iqr


# In[47]:


max_=q3+1.5*iqr
max_


# In[48]:


min_=q1-1.5*iqr
min_


# Summary: There are no outliers as all data points exist within range of max_ and min_

# #### Distribution Plot

# In[49]:


House_data['DIST_MAINROAD'].hist(bins=25)
plt.xlabel('DIST_MAINROAD')
plt.ylabel("Count")
plt.show()


# Summary: The Most houses are at a distance of 75mts from mainroad.

# #### feature vs target

# In[50]:


plt.scatter(House_data['DIST_MAINROAD'],House_data['Total_Price'])
plt.show()


# Summary: Most of the houses sold are within 1.5cr

# ### 5. N_BEDROOM

# #### Check for outliers

# In[51]:


House_data.N_BEDROOM.describe()


# In[52]:


q3,q1=House_data.N_BEDROOM.quantile(0.75),House_data.N_BEDROOM.quantile(0.25)
iqr=q3-q1
iqr


# In[53]:


max_=q3+1.5*iqr
max_


# In[54]:


min_=q1-1.5*iqr
min_


# Summary: we see the max no of bedroom were 4 which is greater than 3.5, so lets check how many data points are outliers

# In[55]:


House_data[House_data['N_BEDROOM']>3.5].shape


# Summary: we have 254 houses among total 7109 with no of bedrooms as 4, so we cannot remove them. we will use them for modelling

# #### Distribution

# In[56]:


House_data['N_BEDROOM'].hist(bins=25)
plt.xlabel('N_BEDROOM')
plt.ylabel("Count")
plt.show()


# Summary: More than 85% of Houses sold are either single bedroom or double bedroom

# #### feature vs target

# In[57]:


House_data.groupby('N_BEDROOM')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: We can see linear relation btw no of bedrooms and Total price, as no of bedrooms increase price increases

# ### 6. N_BATHROOM

# #### Check for outliers

# In[58]:


House_data['N_BATHROOM'].describe()


# In[59]:


q3,q1=House_data.N_BEDROOM.quantile(0.75),House_data.N_BEDROOM.quantile(0.25)
iqr=q3-q1
iqr


# In[60]:


max_=q3+1.5*iqr
max_


# In[61]:


min_=q1-1.5*iqr
min_


# Summary: we cant find any outliers as all data ranges within max_ and min_

# #### Distribution Plot

# In[62]:


House_data['N_BATHROOM'].hist(bins=25)
plt.xlabel('N_BATHROOM')
plt.ylabel("Count")
plt.show()


# Summar: Around 75% of Houses sold were having only one bathroom

# #### Feature vs Target

# In[63]:


House_data.groupby('N_BATHROOM')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: we can see that there is a linear relation btw no of bathroom and Total Price of House

# ### 7. N_ROOM

#  #### Check for outliers

# In[64]:


House_data['N_ROOM'].describe()


# In[65]:


q3,q1=House_data.N_ROOM.quantile(0.75),House_data.N_ROOM.quantile(0.25)
iqr=q3-q1
iqr


# In[66]:


max_=q3+1.5*iqr
max_


# In[67]:


min_=q1-1.5*iqr
min_


# Summary: we can see that there are some outliers as max no of rooms are 6 beyond 5.5

# In[68]:


House_data[House_data['N_ROOM']>5.5].shape


# Summary: We have 254 rows with total rooms 6, so we cannot delete them, we have to use them for model

# #### Distribution Plot

# In[69]:


House_data['N_ROOM'].hist(bins=25)
plt.xlabel('N_ROOM')
plt.ylabel("Count")
plt.show()


# Summary: Most of the houses sold having total no of 4 rooms followed by 3 rooms

# #### feature vs target

# In[70]:


House_data.groupby('N_ROOM')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: We can see that median price value of 5 room house is more than median price value of 6 room house

# ### 8.SALE_COND

# In[71]:


House_data['SALE_COND'].unique()


# Summary: We can see that there are incorrect sale conditions, we have to replace them

# In[72]:


House_data['SALE_COND']=House_data['SALE_COND'].replace(['Ab Normal','PartiaLl','Partiall','AdjLand','Adj Land'],['AbNormal','Partial','Partial','Adjacent Land','Adjacent Land'])


# In[73]:


House_data['SALE_COND'].unique()


# Summary: There are total five categories in sale condition

# #### Distribution Plot`

# In[74]:


House_data['SALE_COND'].hist(bins=25)
plt.xlabel('SALE_COND')
plt.ylabel("Count")
plt.show()


# Summary: The count of houses sold are almost in all sale conditions

# #### Feature vs target

# In[75]:


House_data.groupby('SALE_COND')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: it might seem all sale conditions are of same cost, but there is a linear relation, with the condition, the total price increasing, so we go with label encoding

# In[76]:


## Label Encoding

House_data["SALE_COND"]=House_data["SALE_COND"].replace({"Partial":1,"Normal Sale":2,"Family":3,"AbNormal":4,"Adjacent Land":5})


# ### 9. PARK_FACIL

# In[77]:


House_data['PARK_FACIL'].unique()


# Summary: we can see two categories with same value No and Noo

# In[78]:


House_data["PARK_FACIL"]=House_data["PARK_FACIL"].replace({"Noo":'No'})


# #### Distribution Plot

# In[79]:


House_data['PARK_FACIL'].hist(bins=25)
plt.xlabel('PARK_FACIL')
plt.ylabel("Count")
plt.show()


# Summary: we can see almost equal no of houses sold without parking facility to houses with parking facility

# #### Feature vs Target

# In[80]:


House_data.groupby('PARK_FACIL')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: As it is Binary Categorical feature, we can go ahead with label encoding

# In[81]:


House_data['PARK_FACIL']=House_data['PARK_FACIL'].replace({'No':0,"Yes":1})


# ### 10. YEAR_BUILD

# #### Distribution plot

# In[82]:


House_data['YEAR_BUILD'].hist(bins=25)
plt.xlabel('YEAR_BUILD')
plt.ylabel("Count")
plt.show()


# Summary: We can see most sales happened btw 1980 to 2000

# #### Feature vs Target

# In[83]:


plt.scatter(House_data['YEAR_BUILD'],House_data['Total_Price'])
plt.show()


# Summary: We could see that sales increased over years from 1960 to 2000

# ### 11.BUILDTYPE        

# In[84]:


House_data['BUILDTYPE'].unique()


# In[85]:


House_data['BUILDTYPE']=House_data['BUILDTYPE'].replace({"Comercial":"Commercial","Other":"Others"})


# In[86]:


House_data['BUILDTYPE'].unique()


# Summary: we have corrected the incorrect categories in BUILDTYPE

# #### Distribution

# In[87]:


House_data['BUILDTYPE'].hist(bins=25)
plt.xlabel('BUILDTYPE')
plt.ylabel("Count")
plt.show()


# Summary: we have equal amounts of houses sold for each buildtype

# #### Feature vs Target

# In[88]:


House_data.groupby('BUILDTYPE')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: We can see there is peak difference btw others and commercial, so we cannot do label encoding. we have to do onehot encoding

# In[89]:


House_data=pd.get_dummies(House_data,columns=['BUILDTYPE'])


# ### 12.UTILITY_AVAIL

# In[90]:


House_data['UTILITY_AVAIL'].unique()


# Summary: we can find same categories with incorrect names

# In[91]:


House_data['UTILITY_AVAIL']=House_data['UTILITY_AVAIL'].replace({"All Pub":"AllPub"})


# In[92]:


House_data['UTILITY_AVAIL'].unique()


# Summary: We have corrected the category names

# #### Distribution Plot

# In[93]:


House_data['UTILITY_AVAIL'].hist(bins=25)
plt.xlabel('UTILITY_AVAIL')
plt.ylabel("Count")
plt.show()


# Summary: Houses with ELO utility are less when compared with houses having other three utilities

# #### Feature vs target

# In[94]:


House_data.groupby('UTILITY_AVAIL')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: We cannot find the linear relation btw utility available and Total_price, so we are going with one hot encoding

# In[95]:


House_data=pd.get_dummies(House_data,columns=['UTILITY_AVAIL'])


# ### 13. STREET

# In[96]:


# Check for unique categories available in street

House_data['STREET'].unique()


# Summary: we can see that there are categories repeated with incorrect names

# In[97]:


House_data['STREET']=House_data['STREET'].replace({"Pavd":"Paved","NoAccess":"No Access"}) 


# In[98]:


House_data['STREET'].unique() 


# Summary: we can see now there are only three categories in Street feature

# #### Distribution plot

# In[99]:


House_data['STREET'].hist(bins=25)
plt.xlabel('STREET')
plt.ylabel("Count")
plt.show()


# Summary: We can see most houses sold are under paved, Gravel category than No Access

# #### Feature vs Target 

# In[100]:


House_data.groupby('STREET')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary: Here there is a relation btw street category and total_price of house, however it is non-linear btw No access & Paved, Paved & Gravel. So we have to do one hot encoding

# In[101]:


House_data=pd.get_dummies(House_data,columns=["STREET"])


# ### 14. MZZONE

# In[102]:


# Check for unique categories
House_data['MZZONE'].unique()


# Summary: we have total of 6 unique categories in MZ Zone. 
# * A - Agriculture
# * RH - Residential High Density
# * RL - Residential Low Density
# * I - Industrial
# * C - Commerical
# * RM - Residential Medium Density

# #### Distribution plot
# 

# In[103]:


House_data['MZZONE'].hist(bins=25)
plt.xlabel('MZZONE')
plt.ylabel("Count")
plt.show()


# Summary: From Distribution we can see that Most houses sold are from Residential Areas with all density

# #### Feature vs target

# In[104]:


House_data.groupby('MZZONE')['Total_Price'].median().sort_values(ascending=True).plot.bar()
plt.show()


# Summary1: Mostly people are ready to pay higher prices for Residential Medium than Residential Low density and High Density as they might think of Safety and privacy both

# Summary2: We can see that total price has sudden peak btw Industrial and Residential High density category, so we have to do onehot encoding

# In[105]:


House_data=pd.get_dummies(House_data,columns=['MZZONE'])


# ### 15. QS_ROOMS, QS_BATHROOM, QS_BEDROOM, QS_OVERALL

# #### Distribution plot

# In[106]:


for feature in House_data.columns:
    if feature in ['QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL']:
        House_data[feature].hist(bins=25)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()


# Summary: We can see unusual distribution with other QS_ROOMS, QS_BATHROOM, QS_BEDROOM, but we can find QS_Overall with normal distribution

# #### Feature Vs Target

# In[107]:


for feature in House_data.columns:
    if feature in ['QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL']:
        plt.scatter(House_data[feature],House_data['Total_Price'])
        plt.show()


# Summary: we cannot find any particular relation btw features, so we can check on importance and use these features      

# ### 16. Age_Bldg

# #### Distribution plot

# In[108]:


House_data['Age_Bldg'].hist(bins=25)
plt.xlabel('Age_Bldg')
plt.ylabel("Count")
plt.show()


# Summary: We can most buildings sold are around 10-30 years

# #### Feature vs target

# In[109]:


plt.scatter(House_data['Age_Bldg'],House_data['Total_Price'])
plt.show()


# Summary: In few cases, the prices are high for new buildings and Most buildings sold prices ranged btw 0.7 cr to 1.5 cr

# In[110]:


House_data.shape


# In[111]:


House_data.dtypes


# Summary: we have completed data cleaning and encoding, now we can start with model building

# ### Model Building

# #### Splitting train and test data

# In[112]:


x=House_data.loc[:,House_data.columns!='Total_Price']
y=House_data['Total_Price']


# In[113]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=3)


# #### Linear Regression

# In[114]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Linear Regression Accuracy Score:",model.score(x_test,y_test)*100)


# #### Decision tree

# In[118]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
print("Decision Tree Accuracy Score:",dt.score(x_test,y_test)*100)


# #### Random Forest

# In[124]:


from sklearn.ensemble import RandomForestRegressor

#fit model
rf = RandomForestRegressor(max_depth=18,n_estimators=100,random_state=3)
rf.fit(x_train,y_train)

ypred = rf.predict(x_test)
rf_score = rf.score(x_test,y_test)*100
print("Random Forest score is :",rf_score)


# #### Gradient Boosting

# In[125]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score

#fit model

gradientboosting = GradientBoostingRegressor()
gradientboosting.fit(x_train,y_train)

#predict model

GradientBoosting_score = gradientboosting.score(x_test,y_test)*100
print("Gradient Boosting Accuracy Score:",GradientBoosting_score)


# #### Ada Boost

# In[128]:


from sklearn.ensemble import AdaBoostRegressor

#fit model

adaboost = AdaBoostRegressor(random_state=3,n_estimators=150)
adaboost.fit(x_train,y_train)

#predict model

adaboost_score = adaboost.score(x_test,y_test)*100
print("AdaBoost Accuracy Score:",adaboost_score)


# #### Feature Importance

# In[130]:


feature_scores = pd.Series(rf.feature_importances_,index=x_train.columns).sort_values(ascending=False)
feature_scores=feature_scores*100
feature_scores


# #### Gradient Boost

# In[131]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score


GradientBoosting = GradientBoostingRegressor()
GradientBoosting.fit(x_train,y_train)


GradientBoosting_score = GradientBoosting.score(x_test,y_test)*100
GradientBoosting_score


# ### Summary: of all models we got highest accuracy score of 98.83 from Gradient Boost which is Pretty Best Score 

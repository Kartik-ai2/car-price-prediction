#!/usr/bin/env python
# coding: utf-8

# # It is data of Car Dekho.com downloaded from kaggle .Here it is a used car price prediction dataset
# ## In this we have to predict price of car on the basis of some  features like as torque ,mileage etc
# ## The project is divided into some parts which as follows:
# ## 1)Reading and cleaning dataset
# ## 2)Data Visiualization
# ## 3)Splitting Data into Train and Test
# ## 4)Feature Engineering-Handling Catagorical Features By Encoding Techniques
# ## 5)Feature Selection -Includes correlation, And Extra TreeRegressor
# ## 6)Model Building 
# 

# In[1]:


import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("Car details v3.csv")


# In[4]:


df.isnull().sum()


# ### Droping null values

# In[5]:


df.dropna(inplace=True)
df.reset_index(inplace=True,drop=True)

df.head(15)


# In[6]:



df.corr()


# In[7]:


df.describe()


# ### Data cleaning- converting values into integer by removing string part like in feature mileage  i have split the values by split function and then convert to integer

# In[8]:


def mileage(df):
    for i in range(df["mileage"].shape[0]):
        df["mileage"][i]=df["mileage"][i].split()[0]
    df["mileage(kmpl)"]=df["mileage"]
    df["mileage(kmpl)"]=df["mileage(kmpl)"].astype("float32").astype("int32")

    df.drop("mileage",axis=1,inplace=True)
mileage(df)


# #### Cleaning max_power feature

# In[9]:



def max_power(df):
    df["max_power(bhp)"]=df['max_power']
    
    for i in range(df['max_power'].shape[0]):
        df["max_power(bhp)"][i]=df['max_power(bhp)'][i].split()[0]
            
    df["max_power(bhp)"]=df["max_power(bhp)"].astype("float32").astype('int32')
    df.drop("max_power",axis=1,inplace=True)

    df.head()


max_power(df)


# #### Cleaning engine column

# In[10]:


def engine(df):
    df["engine(cc)"]=df['engine']
    for i in range(df['engine'].shape[0]):
        df["engine(cc)"][i]=df['engine'][i].split()[0]
    df["engine(cc)"]=df["engine(cc)"].astype("float32").astype("int32")
    df.drop("engine",axis=1,inplace=True)

engine(df)


# #### Cleaning name column- breaking name of car to find company of car  

# In[11]:


for i in range(df["name"].shape[0]):
    df["name"][i]=df["name"][i].split()[0]


# #### Cleaning torque feature

# In[12]:


def torque(df):
    
    df["torque(Nm)"]=df["torque"]
    for i in range(df["torque"].shape[0]):
        if "N"in df["torque"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="N")[0]
        if "n"in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="n")[0]
        if "k"in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="k")[0]
        if "K"in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="K")[0]
        if "@"in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="@")[0]
        if " "in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split()[0]     
        if "("in df["torque(Nm)"][i]:
            df["torque(Nm)"][i]=df["torque"][i].split(sep="(")[0]
    df["torque(Nm)"]=df["torque(Nm)"].astype('float32').astype("int32")
    df.drop("torque",axis=1,inplace=True)
        
    
        
torque(df)


# ### Here Below we are checking some liner regression assumption(it means can liner regression give better accuracy or not)

# #### Using QQ plot to find that wheather data is normalized or not--After using QQ plot i found some data is not normally distributed like max_power(bhp)

# In[13]:


from scipy import stats
from scipy.stats import norm

def QQplot(df,variable):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(df[variable],kde=True)
    plt.xlabel(variable)
    plt.subplot(1,2,2)
   




    stats.probplot(df[variable],dist="norm",plot=plt,rvalue=True)
    plt.xlabel(variable)
   
a=["torque(Nm)","max_power(bhp)","km_driven","mileage(kmpl)","year"]
for i in a:
    QQplot(df,i)


# In[14]:


sns.pairplot(df)


# In[15]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.scatter("year","selling_price",data=df)
plt.xlabel("Year")
plt.ylabel("selling_price")
plt.subplot(2,2,2)
plt.scatter("mileage(kmpl)","selling_price",data=df)
plt.ylabel("selling_price")
plt.xlabel("mileage(kmpl)")
plt.subplot(2,2,3)
plt.scatter("seats","selling_price",data=df)
plt.ylabel("selling_price")
plt.xlabel("seats")
plt.subplot(2,2,4)
plt.scatter("max_power(bhp)","selling_price",data=df)
plt.ylabel("selling_price")
plt.xlabel("max_power(bhp)")



# ### year and selling price are highly dependent on each other

# In[16]:


sns.lineplot("seats","selling_price",data=df)


# In[17]:


sns.catplot("seller_type","fuel",data=df,kind="strip")


# 
# 
# ### Some more Data visuilization 

# In[39]:


fig_dims = (16, 7)
fig,ax = plt.subplots(figsize=fig_dims)
sns.histplot(x="name",data=df,color="black")
plt.xticks(rotation=90) 
plt.show()


# #### selling_price vs name and vs owner

# In[36]:


sns.catplot(y='selling_price',x="name",data= df,kind="boxen",height=5, aspect=3)
plt.xticks(rotation=90)
plt.show()


# ### Price is changing with respect to any brand but i think it will not so much useful 

# In[38]:


sns.catplot(y='selling_price',x="owner",data= df,kind="boxen",height=5, aspect=3,hue="fuel")


# #### Here from above graph we have found that  First Owner car and diesel cars have high price 

# In[20]:


sns.catplot(x="fuel", y="selling_price",data=df)


# #### from above graph we found that the LPG and CNG cars price are too low as compare to the Diesel and Petrol

# In[24]:


sns.boxplot(x="transmission",y="selling_price",data=df,hue="fuel")


# ### Automatic Cars have higher price

# ## Splitting Data to train test split

# In[22]:


y=df.iloc[:,2:3]
X=df.drop("selling_price",axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=.25,random_state=250)


# In[23]:


Y_train = Y_train.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)


# In[29]:


Y_train


# # Feature Engineering
# 

# ## Handling Catagorical Features
# ### Handling  Feature-Fuel

# In[30]:


X_train["fuel"].value_counts()
X_train


# 
# #### Here below i have used the Label encoding 

# In[31]:


def fuel(X_train):
    from sklearn.preprocessing import LabelEncoder
    Label=LabelEncoder()
    X=Label.fit_transform(X_train.fuel)
    X_train.fuel=X
    


# In[32]:


fuel(X_train)


# In[33]:


X_train


# ### Handling Feature - transmission

# In[34]:


X_train.transmission.value_counts()


# In[35]:


def transmission(X_train):
    from sklearn.preprocessing import LabelEncoder
    Label=LabelEncoder()
    X=Label.fit_transform(X_train.transmission)
    X_train.transmission=X
  


# In[36]:


transmission(X_train)


# In[37]:






X_train


# ### Handling Feature - owner

# In[38]:


X_train['owner'].value_counts()


# #### Here we can use label encoding like this way mention below but i done it by  by replace function
 
def owner(X_train):
     from sklearn.preprocessing import LabelEncoder
     Label=LabelEncoder()
     X=Label.fit_transform(X_train.owner)
     X_train.owner=X
owner(X_train)
# In[39]:


def owner(X_train):
    X_train.replace({"First Owner":1,"Second Owner":2,"Third Owner": 3,"Fourth & Above Owner":4,"Test Drive Car":5},inplace=True)
X_train


# In[40]:


owner(X_train)


# In[41]:


X_train


# 
# ### Handling Feature - seller_type  -Here i have used label encoding

# In[42]:


def seller_type(X_train):
    from sklearn.preprocessing import LabelEncoder
    Label=LabelEncoder()
    X=Label.fit_transform(X_train.seller_type)
    X_train.seller_type=X


# In[43]:



seller_type(X_train)


# In[44]:


X_train


# ### Handling Feature-name
# #### Here firstly i have split name of company  from the feature name and then i have performed frequency encoding on  name 

# In[45]:


# using frequency encoding
def freq_encoding_on_name(X_train):
    name_dict=X_train.name.value_counts().to_dict()
    X_train["name"]=X_train["name"].map(name_dict)


# In[46]:



freq_encoding_on_name(X_train)


# In[47]:



X_train.head()


# In[48]:


Y_train.iloc[327:329,]


# # Working On test data

# #### Same functions which are works on train data are called below here only parameter are changed X_test is used instead of X_train

# In[51]:


freq_encoding_on_name(X_test)
owner(X_test)
transmission(X_test)
seller_type(X_test)
fuel(X_test)


# ## Feature Selection
# ### By Correlation
# #### Some Data have non linear nature to capture non linear nature i have used spearmen rank correlation
# 

# In[55]:


fig,ax=plt.subplots(figsize=(16,10))
plt.savefig("fig")

sns.heatmap(X_train.corr(method='spearman'),annot=True,cmap='YlOrRd')
# by default the .corr method use pearson correlation ,but here i have used spearman ,because spearman works well on non linear data


# ### By Extra Tree Regressor

# #### Extra Tree Regressor is a Feature selection technique used to select feature which are highly important with respect to target variable

# In[56]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X_train,Y_train)


# In[58]:


model.feature_importances_


# In[59]:


plt.figure(figsize = (12,8))
ranked_features = pd.Series(model.feature_importances_, index=X_train.columns)
ranked_features.nlargest(15).plot(kind='barh')
plt.show()


# #### This above Extra Regressor gives me my important feature according to my target variable

# In[61]:


X_train


# In[62]:


X_train.drop(["name"],axis=1,inplace=True)
X_test.drop(["name"],axis=1,inplace=True)


# In[63]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)
Y_pred= linear_reg.predict(X_test)
print("Accuracy on Training set: ",linear_reg.score(X_train,Y_train))
print("Accuracy on Testing set: ",linear_reg.score(X_test,Y_test))


# In[64]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('Mean Absolute Error=', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared  Error=', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared  Error=', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R Squared Error =', metrics.r2_score(Y_test, Y_pred))


# In[65]:


plt.scatter(Y_test, linear_reg.predict(X_test), color = 'red')
plt.plot(Y_test, linear_reg.predict(X_test), color = 'navy')
plt.title('Linear Regression')
plt.xlabel('Test value')
plt.ylabel('Predicted value')


# ### Here above i have used linear regression but as known it do not gives good accuracy so we will use random forest 

# #### Firstly I have used cross validation 

# In[66]:


from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model =RandomForestRegressor()
# evaluate model
scores = cross_val_score(model, X_train, Y_train, scoring=None, cv=cv, n_jobs=-1)
print(scores)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### Random Forest

# In[67]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=13, min_samples_split=8, n_estimators=150)
rf.fit(X_train, Y_train)
Y_pred=rf.predict(X_test)
print("Accuracy on Training set: ",rf.score(X_train,Y_train))
print("Accuracy on Testing set: ",rf.score(X_test,Y_test))



# In[68]:


print('Mean Absolute Error=', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared  Error=', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared  Error=', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R Squared Error =', metrics.r2_score(Y_test, Y_pred))


# In[69]:


plt.scatter(Y_test, rf.predict(X_test), color = 'red')
plt.plot(Y_test, rf.predict(X_test), color = 'blue')
plt.title('Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Test')
plt.show()


# In[40]:


rf.predict([[2019,7500,1,0,0,1,5.0,16,190,19000,400]])## Actual price for this data is around 5400000 and i got arround 52.7 not soo bad


# #### Below there is a code for hyperparameter tuning which i have used but i have comment down this code because it takes too much time to execute every time
# 
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(0,150,2)]
max_features=["auto","sqrt","log2"]
max_depth=[int(x) for x in np.linspace(0,60,10)]
min_samples_split=[2,3,5,8,9]
min_samples_leaf=[1,2,3,4,6,9]
random_grid={"n_estimators":n_estimators,"max_features":max_features,"max_depth":max_depth,"min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,"criterion":["mse", "mae"]}
print(random_grid)
rf_randomCV=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=25,cv=cv,verbose=2,random_state=50,n_jobs=-1)
rf_randomCV.fit(X_train,Y_train)

rf_randomCV.best_params_rf_randomCV.best_estimator_
# In[70]:


plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(Y_test,Y_pred)
ax.plot(Y_test,Y_pred)

ax.set_xlabel('x')
ax.set_ylabel('y')


# In[71]:


plt.scatter(Y_test,Y_pred)


# ### Using Xgboost

# In[72]:


from xgboost import XGBRegressor
xg_reg=XGBRegressor( learning_rate=0.1, max_depth=6, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=1.0, verbosity=0)


# In[73]:


xg_reg=xg_reg.fit(X_train,Y_train)

Y_pred_model2 = xg_reg.predict(X_test)
print("Accuracy on Training set: ",rf.score(X_train,Y_train))

print("Accuracy on Testing set: ",rf.score(X_test,Y_test))


# In[74]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


print('Mean Absolute Error=', metrics.mean_absolute_error(Y_test, Y_pred_model2))
print('Mean Squared  Error=', metrics.mean_squared_error(Y_test, Y_pred_model2))
print('Root Mean Squared  Error=', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_model2)))
print('R Squared Error =', metrics.r2_score(Y_test, Y_pred_model2))









# In[75]:


value=np.array([[2019,7500,1,0,0,1,5.0,16,190,19000,400]])
xg_reg.predict(value)


# In[76]:


X_train.fuel.unique()


# In[77]:


plt.scatter( Y_test, xg_reg.predict(X_test), color = 'red')
plt.plot(Y_test, xg_reg.predict(X_test), color = 'blue')
plt.title('Xgboost')
plt.xlabel('Predicted')
plt.ylabel('Test')
plt.show()


# In[78]:


import pickle
pickle_model=pickle.dump(xg_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



# In[79]:


list=[[2019,7500,1,0,0,1,5.0,16,190,19000,400]]
a=np.array(list)
model.predict(a)


# In[ ]:





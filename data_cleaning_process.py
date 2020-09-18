#!/usr/bin/env python
# coding: utf-8

# <h1><center><b>Matthew Afsahi</b></center></h1>

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# <h2> Importing Needed Libraries and Reading the Data<h2>

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/Automobile_data.csv')


# In[ ]:


df.info()   # Thre are 16 object types in the data


# In[ ]:


df.head()  # head od the data


# <h1> Data Cleaning and Exploratory Data Analysis</h1>

# In[ ]:


# Importting libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.pairplot(df, hue='num-of-cylinders')  # getting a broad idea of how the data looks like


# In[ ]:


df.corr() # finding the correlations for the numeric values before cleaning the data.


# In[ ]:


# not quite precise before cleaning the data there are many other numerical values that are string type 

fig=plt.figure(figsize=(14,8))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm', linecolor='black')
plt.show()


# In[ ]:


df.describe()


# In[ ]:


sns.lmplot(x='curb-weight',y='width',data=df)


# In[ ]:


# function to transform the string price to numeric vales
def func(text):
  if text=='?':
    return np.nan
  else:
    return int(text)


# In[ ]:


df['price']=df['price'].apply(lambda price:func(price))


# In[ ]:


df['price'].describe()


# In[ ]:


# There are 4 NaN' in price column
df['price'].isnull().sum()


# In[ ]:


df['price'].mean()


# In[ ]:


len(df['price'])


# In[ ]:


# filling the NaN vals with avegarage of the column.
df['price']=df['price'].fillna(df['price'].mean())


# In[ ]:


df['price'].isnull().sum()


# In[ ]:


# price and highway-mpg has a negative slope, which make sense
sns.jointplot(x='highway-mpg',y='price',data=df,kind='reg', color='blue')


# In[ ]:


df['num-of-cylinders'].value_counts()


# In[ ]:


# num-of-cylinders val is a string type it should be a numeric val
type(df['num-of-cylinders'][0])


# In[ ]:


# crating a dictionary to map the number os cylinders
d = {'four':4,
   'for': 4,
   'six': 6,
   'five':5,   
   'eight': 8,
   'two': 2,
   'twelve':12,
   'three':3 }


# In[ ]:


df['num-of-cylinders']=df['num-of-cylinders'].map(d)


# In[ ]:


df.head(6)


# In[ ]:


type(df['num-of-cylinders'][3])


# In[ ]:


df.corr()[['price','num-of-cylinders']] # high corr


# In[ ]:


# EDA to get an idea that how this val is distributed
plt.figure(figsize=(12,6))
sns.countplot(df['num-of-cylinders'])


# In[ ]:


# EDA on length and price vals that it looks a logarithmic relations
plt.figure(figsize=(14,8))
sns.scatterplot(x='length',y='price',hue='body-style',data=df,alpha=.6)


# In[ ]:


# There are two types of the cars in term of the doors with the two missing vals
df['num-of-doors'].value_counts()


# In[ ]:


# function to convert the string type to a numeric type of the doors with taking care of NaN vals
def transform_door(text):
  if text=='?':
    return np.nan
  elif text=='four':
    return (4/1)
  else: 
    return (2/1)


# In[ ]:


df['num-of-doors']=df['num-of-doors'].apply(lambda x:transform_door(x))


# In[ ]:


df['num-of-doors'].value_counts()


# In[ ]:


type(df['num-of-doors'][0])


# In[ ]:


# There are two missing vals on this col, let dro those indexes because the mean of the 2,4 is not meaningfull for a three door car.
df['num-of-doors'].isnull().sum()


# In[ ]:


df['num-of-doors'].dropna(inplace=True)


# In[ ]:


df['num-of-doors'].isnull().sum()


# In[ ]:


# EDA on a box plot shows that there are a lots of outliers on them that could affect, I will taking care of this later below.
plt.figure(figsize=(12,6))
sns.boxplot(x='num-of-doors',y='price',data=df)


# In[ ]:


df.corr()[['price','num-of-doors']] # low corr 


# In[ ]:


# Low corr between price and num-of-doors better to drop this col
df.drop('num-of-doors',axis=1,inplace=True)


# In[ ]:


# EDA with the boxplot visually determins that hardtop cars and convertible cars have higher price than other types of other cars
plt.figure(figsize=(12,6))
sns.boxplot(x='body-style',y='price',data=df)


# In[ ]:


# Viasually can find that there are lots of outlier for the gas type cars in term of their prices
plt.figure(figsize=(12,6))
sns.boxplot(x='fuel-type',y='price',data=df)


# In[ ]:


# EDA on num-of-cylinders how the 2,3 and 12 cylinders are special in our data and also how they are spreaded.
plt.figure(figsize=(12,6))
sns.boxplot(x='num-of-cylinders',y='price',data=df)


# In[ ]:


type(df.symboling[0])


# In[ ]:


type(df['normalized-losses'][0]) # This col is a string type let find out its effect on the data.


# In[ ]:


df['normalized-losses'].value_counts() # there are 41's of missing vals, let clean this and then transform it to an int value


# In[ ]:


# function to convert the normalized-losses val
def clean_normalizedLosses_col(text):
  if text=='?':
    return np.nan
  elif '?' in text:
    return np.nan
  else:
    return int(text)


# In[ ]:


df['normalized-losses']=df['normalized-losses'].apply(lambda text: clean_normalizedLosses_col(text))


# In[ ]:


df['normalized-losses'].isnull().sum()


# In[ ]:


type(df['normalized-losses'][56])


# In[ ]:


# The correlation of the normalized-losses and price is 0.13, so not that high and it could npt be a very important val 
df.corr()[['price','normalized-losses']] 


# In[ ]:


df['normalized-losses'].fillna(df['normalized-losses'].mean(),inplace=True)


# In[ ]:


df['normalized-losses'].value_counts() 


# In[ ]:


# While there is a low correlation between price and normalized-losses let drop this column 
df.drop('normalized-losses',axis=1,inplace=True)


# In[ ]:


df.info()  # There are 12 more objects types to be convert into numeric vals


# In[ ]:


# While the name of the make is not very important, but it may affect on the price, so I will keep the top 10 companies.


# In[ ]:


df.make.value_counts().sort_values(ascending=False).head(10).index


# In[ ]:


top_10=[label for label in df.make.value_counts().sort_values(ascending=False).head(10).index ]


# In[ ]:


top_10


# In[ ]:


# creating a function to encode the string type of make val to numeric vals of 0 and 1's.
def one_hot_top_x(data,col,top_x_labels):

  for label in top_x_labels:
    data[col + '_' + label ]=np.where(data[col]==label,1,0)


# In[ ]:


one_hot_top_x(df,'make',top_10)


# In[ ]:


df.head(2)


# In[ ]:


# Dropping the make col
df.drop('make',axis=1,inplace=True)


# In[ ]:


column_make_factory=[]
for col in df.columns:
  if 'make_' in col:
    column_make_factory.append(col)
column_make_factory


# In[ ]:


# let examin the numerical correlation of the make factories with the price column
# As here the corelations shows that the names of the factory have a low correlation with the prices of the cars
plt.figure(figsize=(16,10))
sns.heatmap(df.corr()[['make_toyota',
 'make_nissan',
 'make_mazda',
 'make_mitsubishi',
 'make_honda',
 'make_volkswagen',
 'make_subaru',
 'make_volvo',
 'make_peugot',
 'make_dodge','price']],yticklabels=['price'],annot=True,cmap='viridis') 


# In[ ]:


# We should dropp the makes columns as long as they have a low or negative corr with the price 
#for reducing number of columns for the sake of not having over fitting on the model

df.drop(columns=column_make_factory,axis=1,inplace=True)


# In[ ]:


df.head(2)


# In[ ]:


# There are 11 vals more to be converted into numeric vals
df.info()


# In[ ]:


df['fuel-type'].value_counts()


# In[ ]:


df['fuel-type'].isnull().sum()


# In[ ]:


# converting the fuel type if gas put 1 else 0
df['fuel-type']=df['fuel-type'].apply(lambda x: 1 if x=='gas' else 0 )


# In[ ]:


df.corr()[['price','fuel-type']] # negative corr


# In[ ]:


for col in df.columns:
  if 'fuel' in col:
    print(col)


# In[ ]:


# very low corr with the price col, so better to drop it
df.drop('fuel-type',inplace=True,axis=1)


# In[ ]:


# Transforming the body-style to the numeric vals which has a high corr with the perice val.
df['body-style'].value_counts().sort_values(ascending=False).index


# In[ ]:


body=[label for label in df['body-style'].value_counts().sort_values(ascending=False).index]


# In[ ]:


body


# In[ ]:


df['body-style'].isnull().sum()


# In[ ]:


one_hot_top_x(df,'body-style',body)


# In[ ]:


df.head(3)


# In[ ]:


df.drop('body-style',axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df['aspiration'].value_counts()


# In[ ]:


type(df['aspiration'][0])


# In[ ]:


df['aspiration'].isnull().sum()


# In[ ]:


df['aspiration']=df['aspiration'].apply(lambda x: 1 if x=='std' else 0)


# In[ ]:


# While there is a negative correlation between aspiration and price I will drop this col
df.drop('aspiration',axis=1,inplace=True)


# In[ ]:


df.head(2)


# In[ ]:


df['drive-wheels'].isnull().sum()


# In[ ]:


df['drive-wheels'].value_counts()


# In[ ]:


# converting the drive-wheels val to a numeric val
df['drive-wheels']=df['drive-wheels'].apply(lambda x: 1 if ((x=='fwd') |( x=='4wd')) else 0)


# In[ ]:


df.corr()[['price','drive-wheels']] # negative correlation


# In[ ]:


# since there is a negative correlation between price and drive-wheel val I will be drop this col
df.drop('drive-wheels',axis=1,inplace=True)


# In[ ]:


df['engine-location'].value_counts()


# In[ ]:


# function to conver this col to a numeric val
df['engine-location']= df['engine-location'].apply(lambda x: 0 if x=='rear[end]' in x.lower() else 1)


# In[ ]:


df['engine-location'].value_counts()


# In[ ]:


df['engine-location'].isnull().sum()


# In[ ]:


df.corr()[['price','engine-location']]   # negative correlation between price col and engine-location


# In[ ]:


# While there is a negative correlation between price val and engine-location val, I will drop this col.
df.drop('engine-location',axis=1,inplace=True)


# In[ ]:


df['wheel-base'].isnull().sum()


# In[ ]:


type(df['wheel-base'][54])


# In[ ]:


df.corr()[['price','wheel-base']]  # there is a high correlation between price and wheel-base col


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr()[['wheel-base','price']],yticklabels=['price','wheel-base'],annot=True,cmap='coolwarm') # .58 high


# In[ ]:


df.length.isnull().sum()


# In[ ]:


type(df['length'][36])


# In[ ]:


df.corr()[['price','length']]  # There is a high correlations with vals in data.


# In[ ]:


df.width.isnull().sum()


# In[ ]:


type(df.width[0])


# In[ ]:


df.corr()[['width','price']]  # There is a high corr with the vals


# In[ ]:


df.height.isnull().sum()


# In[ ]:


type(df.height[22])


# In[ ]:


df.corr()[['price','height']]  # There is a low correlation with two cols


# In[ ]:


# While there is a low corr with the two cols I will drop this col
df.drop('height', axis=1,inplace=True)


# In[ ]:


df['curb-weight'].isnull().sum()


# In[ ]:


type(df['curb-weight'][89])


# In[ ]:


df.corr()[['curb-weight','price']]  # There is a very high corr with the two cols.


# In[ ]:


df['engine-type'].value_counts()


# In[ ]:


# converting the string to numeric vals
top_frequent_type=[label for label in df['engine-type'].value_counts().sort_values(ascending=False).head().index]
top_frequent_type


# In[ ]:


one_hot_top_x(df,'engine-type',top_frequent_type)


# In[ ]:


df.drop('engine-type',axis=1,inplace=True)


# In[ ]:


# Visually it is quite abvious that the correlation between body-style and price is very low, so I am going to drop this col
plt.figure(figsize=(16,10))
sns.heatmap(df.corr()[['body-style_sedan','body-style_hatchback','body-style_wagon','body-style_hardtop','body-style_convertible','price']],yticklabels=['price'],cmap='viridis',annot=True)


# In[ ]:


columns_to_be_droped=[]
for col in df.columns:
  if 'body-style' in col:
    columns_to_be_droped.append(col)

columns_to_be_droped


# In[ ]:


df.drop(columns=columns_to_be_droped,axis=1,inplace=True)


# In[ ]:


df['fuel-system'].isnull().sum()


# In[ ]:


sns.boxplot(x='fuel-system',y='price',data=df)


# In[ ]:


sns.countplot(df['fuel-system'])


# In[ ]:


df.head(2)


# In[ ]:


# While high-way-mpg and city-mpg is the most important for the fuel system of a car I will drop the fuel system kind val from the data.
df.drop('fuel-system',axis=1,inplace=True)


# In[ ]:


df.head(2)


# In[ ]:


df.info() # 4 more objects to be converted


# In[ ]:


df['bore'].isnull().sum()


# In[ ]:


type(df['bore'][29])


# In[ ]:


df['bore'].value_counts()


# In[ ]:


df['bore']=df['bore'].apply(lambda x: np.nan if x=='?' else float(x))


# In[ ]:


df['bore'].isnull().sum()


# In[ ]:


df['bore'].fillna(df['bore'].mean(),inplace=True)


# In[ ]:


df['bore'].isnull().sum()


# In[ ]:


df.corr()[['price','bore']] # There is a high corr with the cols


# In[ ]:


df.info()  # There 3 more string types to be converting


# In[ ]:


df['stroke'].describe()


# In[ ]:


df['stroke'].value_counts()


# In[ ]:


df['stroke']=df['stroke'].apply(lambda x: np.nan if x=='?' else float(x) )


# In[ ]:


df['stroke'].isnull().sum()


# In[ ]:


# filling the miss vals with the mean of the col
df['stroke'].fillna(df['stroke'].mean(),inplace=True)


# In[ ]:


df['stroke'].isnull().sum()


# In[ ]:


df.corr()[['price','stroke']] # very low corr with the two cols


# In[ ]:


# while there is a low correlation with the price and stroke, I will drop this col
df.drop('stroke',axis=1,inplace=True)


# In[ ]:


df['compression-ratio'].value_counts()


# In[ ]:


type(df['compression-ratio'][1])


# In[ ]:


df['compression-ratio'].isnull().sum()


# In[ ]:


df.corr()[['compression-ratio','price']] # There is a low corr with the two cols


# In[ ]:


df.head(1)


# In[ ]:


df.drop('compression-ratio',axis=1,inplace=True)


# In[ ]:


df['horsepower'].value_counts()


# In[ ]:


df['horsepower']=df['horsepower'].apply(lambda x: np.nan if x=='?' else int(x))


# In[ ]:


df['horsepower'].isnull().sum()


# In[ ]:


df['horsepower']=df['horsepower'].fillna(df['horsepower'].mean())


# In[ ]:


df['horsepower'].isnull().sum()


# In[ ]:


df.corr()[['price','horsepower']] # There is a high corr with the two cols


# In[ ]:


df['peak-rpm'].value_counts()


# In[ ]:


df['peak-rpm']=df['peak-rpm'].apply(lambda x: np.nan if x=='?' else int(x))


# In[ ]:


df['peak-rpm'].fillna(df['peak-rpm'].mean(),inplace=True)


# In[ ]:


df['peak-rpm'].isnull().sum()


# In[ ]:


df.corr()[['price','peak-rpm']] # negative corr with the two cols


# In[ ]:


df.drop('peak-rpm',axis=1,inplace=True)


# In[ ]:


df.info()


# <h3> Finding the Correlations </h3>

# In[ ]:


columns_of_engine_type=[]
for col in df.columns:
  if 'engine-type_' in col:
    columns_of_engine_type.append(col)
columns_of_engine_type


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(df.corr()[['engine-type_ohc',
 'engine-type_ohcf',
 'engine-type_ohcv',
 'engine-type_l',
 'engine-type_dohc','price']],annot=True,cmap='coolwarm',yticklabels=['price'])


# In[ ]:


df.corr()[['engine-type_ohc',
 'engine-type_ohcf',
 'engine-type_ohcv',
 'engine-type_l',
 'engine-type_dohc','price']]


# In[ ]:


# it seems just the engine-type_ohcv has a 40% corr with the price val, I will drop the rest of the columns
df.drop(columns=['engine-type_ohc',
 'engine-type_ohcf',
 'engine-type_l',
 'engine-type_dohc'],axis=1,inplace=True)


# In[ ]:


df.corr()[['symboling','price']] # low corr value ~ -0.082


# In[ ]:


df.drop('symboling',inplace=True,axis=1)


# In[ ]:


df.info() # All the columns are numeric with the high correlation with the price col, let double check the data closely.


# <h2> Correlation of Variables </h2>
# <h3> Note: The blue values are very important for this data because of a well car has all the fit features plus less consumption od gas feature. </h3>

# In[ ]:




plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True,cmap='viridis',yticklabels=['price'],linecolor='white',linewidth=.08)


# <h2> Exploratory Data Analysis<h2>
# <h3>All numeric Data clean data</h3>
# 

# In[ ]:


sns.pairplot(df)


# <h3> Clean data all numeric with outliers, without NaN values </h3>

# In[ ]:


# Data has become numeric without null vals
print('DATA INFO-->')
print()
print(df.info())
print('*' * 50)
print('DATA TYPE-->')
print()
print(df.dtypes)
print('*' * 50)
print('DATA IS NULL VALUES -->')
print()
print(df.isnull().sum())
print('*' *50)
print('Data Shape-->')
print(df.shape)
print('*'* 50)
print('Price Column shape-->')
print(df['price'].shape)


# <h2> Exploratory Data Analysis </h2>
# <h3> Data after clening, needs to be scaled, it is not normalized as shown</h3>
# 

# In[ ]:


plt.figure(figsize=(16,8))
sns.set_style('whitegrid')
sns.distplot(df['wheel-base'],bins=30,hist=False,color='red',label='wheel-base')
sns.distplot(df['length'],bins=30,hist=False,color='yellow',label='length')
sns.distplot(df['width'],bins=30,hist=False,color='blue',label='width')
sns.distplot(df['curb-weight'],bins=30,hist=False,color='green',label='curb-weight')
sns.distplot(df['num-of-cylinders'],bins=30,hist=False,color='green',label='num-of-cylinders')
sns.distplot(df['engine-size'],bins=30,hist=False,color='brown',label='engine-size')
sns.distplot(df['bore'],bins=30,hist=False,color='black',label='bore')
sns.distplot(df['horsepower'],bins=30,hist=False,color='cyan',label='horsepower')
sns.distplot(df['city-mpg'],bins=30,hist=False,color='goldenrod',label='city-mpg')
sns.distplot(df['highway-mpg'],bins=30,hist=False,color='lime',label='highway-mpg')
sns.distplot(df['engine-type_ohcv'],bins=30,hist=False,color='magenta',label='engine-type_ohcv')
plt.tight_layout()
plt.title('Data none scaling ')
plt.xlabel('Automobile Dataset')
plt.legend()


# In[ ]:


# standard deviation plot
plt.figure(figsize=(14,7))
df.drop('price',axis=1).std().plot()


# <h2> Scaling Data </h2>

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


data_scaled=scaler.fit_transform(df.drop('price',axis=1))


# In[ ]:


scaled_df=pd.DataFrame(data_scaled,columns=['wheel-base', 'length', 'width', 'curb-weight', 'num-of-cylinders',
       'engine-size', 'bore', 'horsepower', 'city-mpg', 'highway-mpg','engine-type_ohcv'])


# In[ ]:


scaled_df.head()


# <h2> Exploratory Data Analysis </h2>
# <h3> There are lots of outliers, they are noisy and they should be removed </h3>

# In[ ]:


plt.figure(figsize=(16,5))
sns.set_style('whitegrid')
sns.distplot(scaled_df['wheel-base'],bins=20,color='red',label='wheel-base',kde=False,)
sns.distplot(scaled_df['length'],bins=20,color='yellow',label='length',kde=False)
sns.distplot(scaled_df['width'],bins=20,color='blue',label='width',kde=False)
sns.distplot(scaled_df['curb-weight'],bins=20,color='green',label='curb-weight',kde=False)
sns.distplot(scaled_df['num-of-cylinders'],bins=20,color='maroon',label='num-of-cylinders',kde=False)
sns.distplot(scaled_df['engine-size'],bins=20,color='skyblue',label='engine-size',kde=False)
sns.distplot(scaled_df['bore'],bins=20,color='black',label='bore',kde=False)
sns.distplot(scaled_df['horsepower'],bins=20,color='cyan',label='horsepower',kde=False)
sns.distplot(scaled_df['city-mpg'],bins=20,color='goldenrod',label='city-mpg',kde=False)
sns.distplot(scaled_df['highway-mpg'],bins=20,color='lime',label='highway-mpg',kde=False)
sns.distplot(scaled_df['engine-type_ohcv'],bins=20,color='magenta',label='engine-type_ohcv',kde=False)
plt.title('Data scaled Z score')
plt.xlabel('Automobile Dataset')
plt.xlim(-3.5,7)
plt.xticks(np.arange(-3,7,1))
plt.tight_layout()
plt.legend()


# In[ ]:





# <h3> Removing the Z score less than 2 and removing the outliers as well with scaling methods.</h3>

# In[ ]:


from scipy import stats


# In[ ]:


z=np.abs(stats.zscore(scaled_df))


# In[ ]:


np.where(z>2)


# In[ ]:


# This is an outlier.
z[2][6]


# In[ ]:


scaled_df_removed_outliers=scaled_df[(z < 2).all(axis=1)]


# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.distplot(scaled_df_removed_outliers['wheel-base'],bins=20,color='red',label='wheel-base',kde=False,)
sns.distplot(scaled_df_removed_outliers['length'],bins=20,color='yellow',label='length',kde=False)
sns.distplot(scaled_df_removed_outliers['width'],bins=20,color='blue',label='width',kde=False)
sns.distplot(scaled_df_removed_outliers['curb-weight'],bins=20,color='green',label='curb-weight',kde=False)
sns.distplot(scaled_df_removed_outliers['num-of-cylinders'],bins=20,color='maroon',label='num-of-cylinders',kde=False)
sns.distplot(scaled_df_removed_outliers['engine-size'],bins=20,color='skyblue',label='engine-size',kde=False)
sns.distplot(scaled_df_removed_outliers['bore'],bins=20,color='black',label='bore',kde=False)
sns.distplot(scaled_df_removed_outliers['horsepower'],bins=20,color='cyan',label='horsepower',kde=False)
sns.distplot(scaled_df_removed_outliers['city-mpg'],bins=20,color='goldenrod',label='city-mpg',kde=False)
sns.distplot(scaled_df_removed_outliers['highway-mpg'],bins=20,color='lime',label='highway-mpg',kde=False)
sns.distplot(scaled_df_removed_outliers['engine-type_ohcv'],bins=20,color='magenta',label='engine-type_ohcv',kde=False)
plt.title('Data scaled Z score and reduced outliers')
plt.xlabel('Automobile Dataset')
plt.xlim(-2.5,2.6)
plt.xticks(np.arange(-2,3,1))
plt.tight_layout()
plt.legend()


# In[ ]:


scaled_df_removed_outliers.columns


# In[ ]:


plt.figure(figsize=(12,6))

sns.boxplot(scaled_df_removed_outliers['wheel-base'],data=scaled_df_removed_outliers)


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['length'],data=scaled_df_removed_outliers,color='yellow')



# In[ ]:


plt.figure(figsize=(12,6))

sns.boxplot(scaled_df_removed_outliers['width'],data=scaled_df_removed_outliers,color='red')


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['curb-weight'],data=scaled_df_removed_outliers)


# In[ ]:


# It seems after standardized this col it behaves with outliers also, I will drop this col as well.
sns.boxplot(scaled_df_removed_outliers['num-of-cylinders'],data=scaled_df_removed_outliers)


# In[ ]:


scaled_df_removed_outliers.drop('num-of-cylinders',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['engine-size'],data=scaled_df_removed_outliers,color='pink')


# In[ ]:


# There remains two outliers in this col, as long as it is just 2/205 *100= 0.97, which is almost 1%, so I will drop it.
scaled_df_removed_outliers[scaled_df_removed_outliers['engine-size'] >1.9]


# In[ ]:


scaled_df_removed_outliers.drop(index=[15,16],inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['engine-size'],data=scaled_df_removed_outliers,color='pink')


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['bore'],data=scaled_df_removed_outliers,color='cyan')


# In[ ]:


# There is an outlier in the data let remove that
plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['horsepower'],data=scaled_df_removed_outliers,color='brown')


# In[ ]:


scaled_df_removed_outliers[scaled_df_removed_outliers['horsepower'] >1.5]


# In[ ]:


# This horse power is just les than .5 % of data let drop that
scaled_df_removed_outliers.drop(index=[75],inplace=True)


# In[ ]:


# clean col
plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['horsepower'],data=scaled_df_removed_outliers,color='brown')


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['city-mpg'],data=scaled_df_removed_outliers,color='green')


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(scaled_df_removed_outliers['highway-mpg'],data=scaled_df_removed_outliers,color='purple')


# In[ ]:


# since this col is behaved unregularity for this data, let drop that.
sns.boxplot(scaled_df_removed_outliers['engine-type_ohcv'],data=scaled_df_removed_outliers,color='orange')


# In[ ]:


scaled_df_removed_outliers.drop('engine-type_ohcv',inplace=True,axis=1)


# In[ ]:





# <h2> Exploratory Data Analysis </h2>
# <h3> After taking care of the outliers and cleaning all the columns. This is a final data looks like to be exploring<h3>

# In[ ]:


# This is my final data looks like, which it seems, it needs more feature cleanings 
# This data is right skewed, also
plt.figure(figsize=(14,8))
sns.set_style('whitegrid')
sns.distplot(scaled_df_removed_outliers['wheel-base'],bins=20,color='red',label='wheel-base',kde=False,)
sns.distplot(scaled_df_removed_outliers['length'],bins=20,color='yellow',label='length',kde=False)
sns.distplot(scaled_df_removed_outliers['width'],bins=20,color='blue',label='width',kde=False)
sns.distplot(scaled_df_removed_outliers['curb-weight'],bins=20,color='green',label='curb-weight',kde=False)
#sns.distplot(scaled_df_removed_outliers['num-of-cylinders'],bins=20,color='maroon',label='num-of-cylinders',kde=False)
sns.distplot(scaled_df_removed_outliers['engine-size'],bins=20,color='skyblue',label='engine-size',kde=False)
sns.distplot(scaled_df_removed_outliers['bore'],bins=20,color='black',label='bore',kde=False)
sns.distplot(scaled_df_removed_outliers['horsepower'],bins=20,color='cyan',label='horsepower',kde=False)
sns.distplot(scaled_df_removed_outliers['city-mpg'],bins=20,color='goldenrod',label='city-mpg',kde=False)
sns.distplot(scaled_df_removed_outliers['highway-mpg'],bins=20,color='lime',label='highway-mpg',kde=False)
#sns.distplot(scaled_df_removed_outliers['engine-type_ohcv'],bins=20,color='magenta',label='engine-type_ohcv',kde=False)
plt.title('Data scaled Z score and reduced outliers')
plt.xlabel('Automobile Dataset')
plt.xlim(-2.5,2.6)
plt.xticks(np.arange(-2,3,1))
plt.tight_layout()
plt.legend()


# <h2> Summary of a clean data after removing outliers</h2>

# In[ ]:


print('FINAL DATA Features of my Solution-------------------------->')
print()
print('*' * 100)
print()
print('INFO')
print(scaled_df_removed_outliers.info())
print('*' * 100)
print()
print('Standard Deviation')
print(scaled_df_removed_outliers.std())
print('*' * 100)
print()
print('Description')
print(scaled_df_removed_outliers.describe())
print('*' * 100)
print()
print('Data Shape')
print(scaled_df_removed_outliers.shape)


# <h3>Moving the old price column to the new clean data requires the indexes be matched </h3>

# In[ ]:


price=pd.DataFrame(df['price'],index=scaled_df_removed_outliers.index)


# In[ ]:


price.shape


# In[ ]:


price.isnull().sum()


# In[ ]:


if (scaled_df_removed_outliers.index.equals(price.index)):
  print(' The indexes are equal')


# <h3> Price column is a target column in this notebook, matching the indexes of the two different datasets are important</h3>

# In[ ]:


scaled_df_removed_outliers['new_price_to_be_tested']=price


# In[ ]:


scaled_df_removed_outliers.head()


# In[ ]:


df.head()


# In[ ]:


# succesfully added the price col to scaled clean data frame.


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X= scaled_df_removed_outliers.drop('new_price_to_be_tested',axis=1) # feature cols
y=scaled_df_removed_outliers['new_price_to_be_tested']  # target col


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# <h3>Training the Model</h3>

# In[ ]:


from sklearn.linear_model import LinearRegression


# <h3>instance of a LinearRegression() </h3>

# In[ ]:


model=LinearRegression()


# In[ ]:


model.fit(X_train,y_train)


# <h3> coefficients of the model </h3>

# In[ ]:


print('Coefficient: \n', model.coef_)


# <h3>Predicting Test Data</h3>
# 

# In[ ]:


predictions=model.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)
plt.xlabel(' Y Test')
plt.ylabel('Predicted Y')


# Evaluating the Model
# 

# In[ ]:


from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# **Residuals**
# 
# Shape of the residuals distribution is almost normal distribution, but it needs more feature engineering.

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot((y_test-predictions),bins=20,color='blue')
plt.xlabel('Residual Distribution')


# **Conclusion**

# In[ ]:


coeffecients=pd.DataFrame(model.coef_,X.columns)
coeffecients.columns=['Coeffecient']
coeffecients


# Intepreting the coeffecients:
# Hollding all other features fixed, a 1 unit increase in wheel-base is associated with an increase of 2199 total dollars spent. Also, like that a 1 unit increse in horsepower associatted with an increase of 2152 total dollars spent. Therefore, focussing on the variables wheel-base,curb-weight,engine-size,horsepower and highway-mpg is more likely beneficial for a car factory.


# Module 1 Project  - King County House Sales 


### Matt Sparr - Self-Paced Data Science Program - Review 10/31/2018

## Introduction
<h4> The purpose of this project is to investigate the variables of homes sold in King County, Washington and to develop a model to accurately predict home sale prices. We will be following the OSEMiN model which involves the steps of obtain, scrub, explore, model, and finally interpret. 

## Obtaining the Data
<h4> We will first import all the necessary libaries and modules and then use Pandas to grab the dataset from the kc_house_data.csv and store it as a Pandas Dataframe.



```python
#importing required libraries and setting matplotlib to inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as lm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from statsmodels.formula.api import ols

%matplotlib inline
```


```python
#grabbing the dataset
df = pd.read_csv('kc_house_data.csv')
```

## Scrubbing the Data
<h4> Now we will clean the data to enable us to create our model without errors. First we will drop the 'id', 'date', 'lat', 'long', and 'zipcode' columns as they will not be needed. We will then inspect the dataframe and address the columns will null or missing values.


```python
#dropping id and date columns
df.drop(['id', 'date', 'lat', 'long', 'zipcode'], axis=1, inplace=True)
```


```python
#viewing the first 15 rows
df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1230000.0</td>
      <td>4</td>
      <td>4.50</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>3890</td>
      <td>1530.0</td>
      <td>2001</td>
      <td>0.0</td>
      <td>4760</td>
      <td>101930</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1715</td>
      <td>?</td>
      <td>1995</td>
      <td>0.0</td>
      <td>2238</td>
      <td>6819</td>
    </tr>
    <tr>
      <th>7</th>
      <td>291850.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1060</td>
      <td>9711</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>7</td>
      <td>1060</td>
      <td>0.0</td>
      <td>1963</td>
      <td>0.0</td>
      <td>1650</td>
      <td>9711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>229500.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1780</td>
      <td>7470</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1050</td>
      <td>730.0</td>
      <td>1960</td>
      <td>0.0</td>
      <td>1780</td>
      <td>8113</td>
    </tr>
    <tr>
      <th>9</th>
      <td>323000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1890</td>
      <td>6560</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1890</td>
      <td>0.0</td>
      <td>2003</td>
      <td>0.0</td>
      <td>2390</td>
      <td>7570</td>
    </tr>
    <tr>
      <th>10</th>
      <td>662500.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>3560</td>
      <td>9796</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1860</td>
      <td>1700.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>2210</td>
      <td>8925</td>
    </tr>
    <tr>
      <th>11</th>
      <td>468000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1160</td>
      <td>6000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>860</td>
      <td>300.0</td>
      <td>1942</td>
      <td>0.0</td>
      <td>1330</td>
      <td>6000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>310000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1430</td>
      <td>19901</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1430</td>
      <td>0.0</td>
      <td>1927</td>
      <td>NaN</td>
      <td>1780</td>
      <td>12697</td>
    </tr>
    <tr>
      <th>13</th>
      <td>400000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1370</td>
      <td>9680</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>7</td>
      <td>1370</td>
      <td>0.0</td>
      <td>1977</td>
      <td>0.0</td>
      <td>1370</td>
      <td>10208</td>
    </tr>
    <tr>
      <th>14</th>
      <td>530000.0</td>
      <td>5</td>
      <td>2.00</td>
      <td>1810</td>
      <td>4850</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1810</td>
      <td>0.0</td>
      <td>1900</td>
      <td>0.0</td>
      <td>1360</td>
      <td>4850</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking value in sqft_basement
df.sqft_basement.value_counts()
```




    0.0       12826
    ?           454
    600.0       217
    500.0       209
    700.0       208
    800.0       201
    400.0       184
    1000.0      148
    300.0       142
    900.0       142
    200.0       105
    750.0       104
    530.0       103
    480.0       103
    450.0       103
    720.0        98
    620.0        90
    580.0        84
    840.0        83
    420.0        81
    860.0        79
    1100.0       78
    670.0        78
    550.0        76
    780.0        76
    650.0        75
    240.0        74
    680.0        73
    380.0        73
    850.0        72
              ...  
    946.0         1
    20.0          1
    276.0         1
    2850.0        1
    666.0         1
    3500.0        1
    1284.0        1
    2400.0        1
    2810.0        1
    1548.0        1
    243.0         1
    784.0         1
    2610.0        1
    2600.0        1
    172.0         1
    143.0         1
    274.0         1
    1275.0        1
    704.0         1
    508.0         1
    861.0         1
    2120.0        1
    556.0         1
    862.0         1
    65.0          1
    792.0         1
    374.0         1
    2196.0        1
    1920.0        1
    2180.0        1
    Name: sqft_basement, Length: 304, dtype: int64



<h4>There appears to be some values of '?' in sqft_basement. We will replace these with the column mean.</h4>


```python
#replacing missing data in sqft_basement with column mean
sqft_basement_numerical = df.sqft_basement[df.sqft_basement != '?']
sqft_basement_numerical = sqft_basement_numerical.astype(float)
sqft_basement_numerical_mean = round(sqft_basement_numerical.mean(),1)
df['sqft_basement'] = (df['sqft_basement'].map(lambda x: sqft_basement_numerical_mean if x == '?' else x)).astype(float)
```


```python
#checking for null values
df.isna().sum()
```




    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



<h4>The 'view', 'waterfront', and 'yr_renovated' columns all have null values. We will assume that those homes had no views, are not waterfront, and/or were not renovated and fill those null values with 0.


```python
#replacing null values in waterfront with 0
df['view'] = df['view'].fillna(0)
df['waterfront'] = df['waterfront'].fillna(0)
df['yr_renovated'] = df['yr_renovated'].fillna(0)
```

<h4> Now before we modify any of the columns we will create a copy of the dataframe to give us an unaltered version to run models on later for comparison. 


```python
#creating copy of df before transformations
df_original = df.copy()
```

## Exploring the Data
<h4> For this section, we will begin by inspecting the histogram of each variable to look at the distributions and skew.


```python
#intial histogram of all variables
df.hist(figsize=(15,15))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80546F60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F8045E978>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80492048>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F804B86D8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F804DED68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F804DEDA0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80A5A940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80A81FD0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80AB26A0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80ADBD30>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80B0B400>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80B32A90>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80B64160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80B8B7F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80BB5E80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F80BE5550>]],
          dtype=object)




![png](student2_files/student2_18_1.png)


<h4> An initial observation of the histograms reveals that nearly none of the variables exhibit a normal distribution and there are outliers in many of the variables. To address these issues, we will iterate through each variable and perform adjustments and transformations in order to give us a more normal distribution.


```python
#checking bedrooms for outliers
df.bedrooms.value_counts()
```




    3     9824
    4     6882
    2     2760
    5     1601
    6      272
    1      196
    7       38
    8       13
    9        6
    10       3
    11       1
    33       1
    Name: bedrooms, dtype: int64




```python
#histogram before removal of outliers
df.bedrooms.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f8149d0f0>




![png](student2_files/student2_21_1.png)



```python
#removing outliers
df.drop(df.index[df['bedrooms'] >= 8], inplace=True)
```


```python
#histogram after removal of outliers
df.bedrooms.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80db9898>




![png](student2_files/student2_23_1.png)


<h6> The 'bedrooms' column had some outliers so all homes with more than 7 bedrooms were removed from the dataframe. The number of homes removed was 24 which represents an insignificant percentage of our total data so removing them should not significantly impact the final model. 


```python
#bathrooms histogram
df.bathrooms.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80553eb8>




![png](student2_files/student2_25_1.png)



```python
#using log transformation on bathrooms
(np.log(df.bathrooms)).hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80605b38>




![png](student2_files/student2_26_1.png)



```python
#transforming bathrooms
df.bathrooms = np.log(df.bathrooms)
```

<h6> The 'bathrooms' column showed a negatively skewed distribution so we transformed it using a log transformation as the data now exhibits a more normal distribution.


```python
#sqft_living histogram
df.sqft_living.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80686d68>




![png](student2_files/student2_29_1.png)



```python
#checking count of outliers
print(len(df[df.sqft_living > 8000]))
print(len(df[df.sqft_living > 6000]))
print(len(df[df.sqft_living > 4000]))
```

    9
    67
    773
    


```python
#removing outliers
df.drop(df.index[df['sqft_living'] > 4000], inplace=True)
```


```python
#histogram after removal of outliers
df.sqft_living.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f810d5048>




![png](student2_files/student2_32_1.png)



```python
#histogram of square root transformed sqft_living
(np.sqrt(df.sqft_living)).hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80e12198>




![png](student2_files/student2_33_1.png)



```python
#square root transformation on sqft_living
df.sqft_living = np.sqrt(df.sqft_living)
```

<h6> The 'sqft_living' column had some significant outliers. Homes with more than 4000 square feet were removed which was a little over 800 entries. The histogram of the modified data still showed some negative skew so we then performed a square root transformation thus giving us a more normal distribution. 


```python
#sqft_above histogram
df.sqft_above.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80e72668>




![png](student2_files/student2_36_1.png)



```python
#square root transformation
df.sqft_above = np.sqrt(df.sqft_above)
```


```python
#square root transformed histogram
df.sqft_above.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f8101b710>




![png](student2_files/student2_38_1.png)


<h6> The 'sqft_above' column showed a negative skew so we transformed it using a square root transformation to make the distribution more normal.


```python
#sqft_basement histogram
df.sqft_basement.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80eee588>




![png](student2_files/student2_40_1.png)



```python
#creating new column that declares whether or not a home has a basement
df['basement'] = df.apply(lambda row: 0 if row.sqft_basement == 0 else 1, axis=1)
```


```python
#new binary basement histogram
df.basement.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80f6b198>




![png](student2_files/student2_42_1.png)


<h6> The 'sqft_basement' column has a lot of values of 0 which heavily skews the distribution. By creating a new column 'basement' which is binary in nature and declares whether or not a home has a basement, we will better be able to see how this variable affects the home price later on in our model.


```python
#viewing stats of yr_built
df.yr_built.describe()
```




    count    20800.000000
    mean      1970.352644
    std         29.308936
    min       1900.000000
    25%       1951.000000
    50%       1973.000000
    75%       1995.000000
    max       2015.000000
    Name: yr_built, dtype: float64




```python
#creating new column age that declares the age of the home
df['age'] = (df.yr_built.max() - df.yr_built + 1)
```


```python
#histogram of new age column
df.age.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f811e3ef0>




![png](student2_files/student2_46_1.png)



```python
#square root transformation on age
df.age = np.sqrt(df.age)
df.age.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f827f81d0>




![png](student2_files/student2_47_1.png)



```python
#removal of old yr_built column
df.drop('yr_built',axis=1,inplace=True)
```

<h6> The 'yr_built' column did not make a lot of sense to have in its current state as it represents the age of the home but in terms of the year built. To remedy this we created a new column 'age' which subtracts the year the home was built from the most recent year in the data - 2015. A value of 1 was also added to the age in order to allow us to transform it without error.


```python
#sqft_lot histogram
df.sqft_lot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82835e80>




![png](student2_files/student2_50_1.png)



```python
#checking number of outliers
len(df[df.sqft_lot>200000])
```




    213




```python
#removal of outliers
df.drop(df.index[df['sqft_lot'] > 200000], inplace=True)
```


```python
#histogram after removal of outliers
df.sqft_lot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82bc9ac8>




![png](student2_files/student2_53_1.png)



```python
#log transformation of sqft_lot
df.sqft_lot = np.log(df.sqft_lot)
df.sqft_lot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f80f60400>




![png](student2_files/student2_54_1.png)


<h6> For the 'sqft_lot' column, we first had to remove some of the outliers (213 values) and then performed a log transformation on the data to normalize the distribution.


```python
#histogram of sqft_lot15
df.sqft_lot15.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f8295ddd8>




![png](student2_files/student2_56_1.png)



```python
#log transformed sqft_lot15
df.sqft_lot15 = np.log(df.sqft_lot15)
df.sqft_lot15.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f829c3198>




![png](student2_files/student2_57_1.png)


<h6> Similarly to the 'sqft_lot' column, we performed a log transformation on the 'sqft_lot15' column to normalize the distribution. 


```python
#view histogram
df.view.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82a389e8>




![png](student2_files/student2_59_1.png)



```python
#unique values of view
df.view.unique()
```




    array([0., 3., 4., 2., 1.])




```python
#converting all view values above 0 to 1 to signify that the home has been viewed
df['viewed'] = df.view
i = 4
while i > 0:
    df.viewed = df.viewed.replace(i, 1)
    i -= 1
df.viewed.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82aacc18>




![png](student2_files/student2_61_1.png)


<h6> The 'view' column only contains values 0, 1, 2, 3 and 4 with most of the values being 0. To better represent this data, an additional column was added which tells us whether or not a home was view, represented by a 1 or 0. 


```python
#price histogram
df.price.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82b2e240>




![png](student2_files/student2_63_1.png)



```python
#log transformation on price
df['price_log'] = np.log(df.price)
df.price_log.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f82deafd0>




![png](student2_files/student2_64_1.png)


<h6> Finally, we perform a log transformation on our dependent variable 'price' in order to normalize its distribution. 


```python
#updated histogram of all variables after transformations and adjustments
df.hist(figsize=(14,14))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82DB2AC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82E9BC88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82C2D358>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82C539E8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82C870B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82C870F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82CD6DD8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82D0A470>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82D31B00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82D651D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82D89860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82ED4EF0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82F035C0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82F2DC50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82F5F320>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82F879B0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82FB8080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F82FE1710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F8300ADA0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018F8303A470>]],
          dtype=object)




![png](student2_files/student2_66_1.png)


<h6> Above we can see all of the updated histograms for our variables. Many of the distributions are now normal apart from our created binary columns and their original counterparts.

 

## Model the Data
<h4> Now it is time to create a model for our data. We will be using the function 'stepwise_selection' below to rank the predictors of our data on their contribution to the model and to find the best number of predictors to use.


```python
#stepwise selection function
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
```

<h6> We will now split our data into a train and test set using an initial test size of 0.25. We will then use the stepwise function above to do an interative feature selection for all of the independent variables. For each iteration, one more predictor will be selected and added, and in addition, a repeated k-fold with 3 splits and 10 repeats will be ran to cross validate the R-squared value and % difference in MSE between the train and test set for each regression.
Then, by graphing a plot of our predicted y-values vs the actual y-values and comparing the R-squared values and % differences in MSE, we should be able to decide which predictors to include in our model of best fit. 


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df.drop(['price', 'price_log'], axis=1)
y = df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df.drop(['price', 'price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    #creating a plot of predicted values vs actual values
    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors")
    ax.legend()
    i += 1
results_df['# Predictors'] = predictors
results_df['R-squared'] = reg_score
results_df['MSE % Difference'] = mse_diffs
results_df['Added Predictor'] = added_pred
plt.show()
results_df
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_72_1.png)



![png](student2_files/student2_72_2.png)



![png](student2_files/student2_72_3.png)



![png](student2_files/student2_72_4.png)



![png](student2_files/student2_72_5.png)



![png](student2_files/student2_72_6.png)



![png](student2_files/student2_72_7.png)



![png](student2_files/student2_72_8.png)



![png](student2_files/student2_72_9.png)



![png](student2_files/student2_72_10.png)



![png](student2_files/student2_72_11.png)



![png](student2_files/student2_72_12.png)



![png](student2_files/student2_72_13.png)



![png](student2_files/student2_72_14.png)



![png](student2_files/student2_72_15.png)



![png](student2_files/student2_72_16.png)



![png](student2_files/student2_72_17.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.056778</td>
      <td>2.397657</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.438552</td>
      <td>0.852032</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.468010</td>
      <td>2.275153</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.471852</td>
      <td>2.163946</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.502580</td>
      <td>1.729208</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.541981</td>
      <td>0.993599</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.550919</td>
      <td>0.580638</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.555629</td>
      <td>0.476402</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.558263</td>
      <td>0.380806</td>
      <td>[viewed]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.561765</td>
      <td>0.260768</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.578160</td>
      <td>-0.003382</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.593160</td>
      <td>-0.228657</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.595386</td>
      <td>-0.267875</td>
      <td>[basement]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.596291</td>
      <td>-0.293498</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.599252</td>
      <td>-0.477339</td>
      <td>[sqft_basement]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.613704</td>
      <td>-1.048700</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.614463</td>
      <td>-0.943803</td>
      <td>[yr_renovated]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> From the above table, we can see that the test set MSE is anywhere from approx 1% more to 1% less than the training set MSE. This varies slightly as more predictors are added and doesn't follow a pattern unlike the R-squared values which increases with each additional predictor. Looking at the generated plots of measured vs predicted values for 'price', we can see that as more predictors are added, the predicted values approach the measured values. Based on these findings, we will use all 17 predictors in our model as that will maximize our R-squared value of the model against our test data and also will maintain a moderately low difference in MSE between the train and test set. 

## Evaluate the Model
<h4> Now that we have our model, we can begin to ask relevant questions as to the effectiveness and fit of the model as well as address concerns and investigate possible improvements.

## Question 1
### How does our model compare to a model of the data before any of our modifications and transformations?

<h6> Before we started to dig into and modify our data set, we grabbed a copy of an unedited version of the data - 'df_orig'. We can run a regression on that dataset and compare the results to those of our model to answer this question. We will repeat the stepwise feature selection as we did earlier, using a repeated k-fold validation method, and plotting the predicted values vs the actual values.


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_original.drop(['price'], axis=1)
y = df_original.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_original = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_original.drop('price', axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors")    
    ax.legend()
    i += 1
results_df_original['# Predictors'] = predictors
results_df_original['R-squared'] = reg_score
results_df_original['MSE % Difference'] = mse_diffs
results_df_original['Added Predictor'] = added_pred
plt.show()
results_df_original
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_77_1.png)



![png](student2_files/student2_77_2.png)



![png](student2_files/student2_77_3.png)



![png](student2_files/student2_77_4.png)



![png](student2_files/student2_77_5.png)



![png](student2_files/student2_77_6.png)



![png](student2_files/student2_77_7.png)



![png](student2_files/student2_77_8.png)



![png](student2_files/student2_77_9.png)



![png](student2_files/student2_77_10.png)



![png](student2_files/student2_77_11.png)



![png](student2_files/student2_77_12.png)



![png](student2_files/student2_77_13.png)



![png](student2_files/student2_77_14.png)



![png](student2_files/student2_77_15.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.100510</td>
      <td>-0.198471</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.490020</td>
      <td>4.409659</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.515821</td>
      <td>5.126805</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.525980</td>
      <td>6.090957</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.540883</td>
      <td>5.089502</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.543076</td>
      <td>5.013684</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.600565</td>
      <td>4.493230</td>
      <td>[yr_built]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.609256</td>
      <td>4.418562</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.630056</td>
      <td>4.366307</td>
      <td>[sqft_basement]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.650623</td>
      <td>4.656122</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.650965</td>
      <td>4.607029</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.650560</td>
      <td>4.989960</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.650894</td>
      <td>4.902002</td>
      <td>[yr_renovated]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.652061</td>
      <td>5.093368</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.652053</td>
      <td>5.091840</td>
      <td>[sqft_lot]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> Comparing the regression of the data in 'df_original' to that of our model we can see that similarly, the R-squared value of the test set increases as the number of predictors increase apart from the last step in going from 13 to 14 predictors where the R-squared value slightly decreases. The R-squared of our model using 17 predictors was 0.614463 while the best R-squared of the un-transformed dataset is actually higher at 0.652061 using 13 predictors. This is interesting as based on that metric alone, the model of 'df_original' actually fits the test set slightly better on average. Conversely, however, the % difference in MSE between the train and test set is lower in our model with it being -0.943803 and for 'df_original' using 13 predictors, 5.093368.  
    When we look at the plots of predicted vs measured values, we can see that the model of 'df_original' fits the data moderately well as more predictors are added. However, the model of 'df_original' starts to do a worse job of predicting as the 'price' increases.
    It would appear that our model does fit the data better and is a better predictor of home prices, but to a lesser extent than would be expected. To investigate why this might be the case, we can ask further questions.


## Question 2
### Can our model be improved by addressing covariance of the predictors?

<h6> If multiple variables exhibit covariance with each other, our model could be affected negatively. We will check the covariance of all of our predictors and try eliminating problematic variables to see if it improves our model.


```python
#heatmap to check for correlations between variables where corr > 0.75
sns.heatmap(df.corr(),center=0);
```


![png](student2_files/student2_81_0.png)


<h6> Some of our variables appear to be highly correlated. We will choose a threshold of 0.75 for variables with significant correlation and generate a new heatmap to more clearly show those variables.


```python
sns.heatmap(abs(df.corr())>0.75, center=0)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f87a62b00>




![png](student2_files/student2_83_1.png)


<h6> From the above we can see that 'sqft_above' and 'sqft_living', 'sqft_lot15' and 'sqft_lot', 'view' and 'viewed', and 'sqft_basement' and 'basement', show high correlation with each other. Out of each of these pairs, the less important ones are 'sqft_above' as the total square footage of a home is likely more significant, 'sqft_lot', 'viewed', and 'basement'. Now let's try running our regression after removing these variables.


```python
#creating dataframe that will remove some of the variables with high correlation to other variables
df_adj = df.copy()
to_drop = ['view', 'sqft_lot15', 'sqft_above', 'sqft_basement']
df_adj.drop(to_drop, axis=1, inplace=True)
```


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_adj.drop(['price','price_log'], axis=1)
y = df_adj.price_log
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_adj = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_adj.drop(['price','price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors")
    ax.legend()
    i += 1
results_df_adj['# Predictors'] = predictors
results_df_adj['R-squared'] = reg_score
results_df_adj['MSE % Difference'] = mse_diffs
results_df_adj['Added Predictor'] = added_pred
plt.show()
results_df_adj
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_86_1.png)



![png](student2_files/student2_86_2.png)



![png](student2_files/student2_86_3.png)



![png](student2_files/student2_86_4.png)



![png](student2_files/student2_86_5.png)



![png](student2_files/student2_86_6.png)



![png](student2_files/student2_86_7.png)



![png](student2_files/student2_86_8.png)



![png](student2_files/student2_86_9.png)



![png](student2_files/student2_86_10.png)



![png](student2_files/student2_86_11.png)



![png](student2_files/student2_86_12.png)



![png](student2_files/student2_86_13.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.036134</td>
      <td>2.199824</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.359981</td>
      <td>0.310695</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.465546</td>
      <td>-0.247919</td>
      <td>[viewed]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.489933</td>
      <td>-0.641431</td>
      <td>[basement]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.494739</td>
      <td>-0.898407</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.495767</td>
      <td>-0.913714</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.567101</td>
      <td>-2.008907</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.569948</td>
      <td>-2.013282</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.570385</td>
      <td>-2.006357</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.571658</td>
      <td>-2.188099</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.606146</td>
      <td>-3.098479</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.619440</td>
      <td>-3.508948</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.619650</td>
      <td>-3.433841</td>
      <td>[yr_renovated]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> Looking at the results of this regression, we can see that the R-squared value and % difference in MSE are approximately the same in this adjusted model as in our original model. This adjusted model is a slightly worse fit as the R-squared values are slightly lower and the % differences in MSE are slightly more. This seems to suggest that the covariance of the removed variables does not significantly affect our model's fit in a negative way. One additional thing to consider however, is our added variables of basement and viewed. Both of which are binary data columns meant to deal with both columns having a large number of 0s and then a small spread of other values. We will try running a regression simply without those two added variables to see if that makes any difference. 


```python
#creating dataframe that will remove some of the variables with high correlation to other variables
df_adj2 = df.copy()
to_drop = ['viewed', 'basement']
df_adj2.drop(to_drop, axis=1, inplace=True)
```


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_adj2.drop(['price', 'price_log'], axis=1)
y = df_adj2.price_log
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_adj2 = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_adj2.drop(['price', 'price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors")
    ax.legend()
    i += 1
results_df_adj2['# Predictors'] = predictors
results_df_adj2['R-squared'] = reg_score
results_df_adj2['MSE % Difference'] = mse_diffs
results_df_adj2['Added Predictor'] = added_pred
plt.show()
results_df_adj2
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_89_1.png)



![png](student2_files/student2_89_2.png)



![png](student2_files/student2_89_3.png)



![png](student2_files/student2_89_4.png)



![png](student2_files/student2_89_5.png)



![png](student2_files/student2_89_6.png)



![png](student2_files/student2_89_7.png)



![png](student2_files/student2_89_8.png)



![png](student2_files/student2_89_9.png)



![png](student2_files/student2_89_10.png)



![png](student2_files/student2_89_11.png)



![png](student2_files/student2_89_12.png)



![png](student2_files/student2_89_13.png)



![png](student2_files/student2_89_14.png)



![png](student2_files/student2_89_15.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.022288</td>
      <td>2.036337</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.442665</td>
      <td>-0.313025</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.451988</td>
      <td>-0.735181</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.462785</td>
      <td>-0.583648</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.523378</td>
      <td>-1.350915</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.557287</td>
      <td>-1.814305</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.560180</td>
      <td>-1.840078</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.562740</td>
      <td>-1.745170</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.564594</td>
      <td>-1.998607</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.564573</td>
      <td>-1.995191</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.605012</td>
      <td>-2.953431</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.605242</td>
      <td>-2.977561</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.618687</td>
      <td>-3.311335</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.618789</td>
      <td>-3.266214</td>
      <td>[yr_renovated]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.618900</td>
      <td>-3.232481</td>
      <td>[sqft_basement]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> Again, it seems that the fit is similar in goodness to our model but slightly worse as marked by slightly higher R-squared values and slightly more % difference in MSE between the test and train sets. For this reason we will keep our model as is, including our created columns of 'basement' and 'viewed'.

## Question 3
### Can we make a better fitted model by selecting a subset of homes within a certain price range?

<h6> Let's begin by viewing the statistics of 'price' and its distribution.


```python
df.price.describe()
```




    count    2.058700e+04
    mean     5.023165e+05
    std      2.727140e+05
    min      7.800000e+04
    25%      3.155000e+05
    50%      4.400000e+05
    75%      6.150000e+05
    max      3.100000e+06
    Name: price, dtype: float64




```python
df.price.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18f88c14438>




![png](student2_files/student2_94_1.png)


<h6> 75% of homes have a 'price' of 615,000 or less. Let's try using this as our subset and seeing how that affects our generated models.


```python
df_s = df[df.price <= 615000]
```


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_s.drop(['price', 'price_log'], axis=1)
y = df_s.price_log
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_s = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_s.drop(['price', 'price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors") 
    ax.legend()
    i += 1
results_df_s['# Predictors'] = predictors
results_df_s['R-squared'] = reg_score
results_df_s['MSE % Difference'] = mse_diffs
results_df_s['Added Predictor'] = added_pred
plt.show()
results_df_s
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_97_1.png)



![png](student2_files/student2_97_2.png)



![png](student2_files/student2_97_3.png)



![png](student2_files/student2_97_4.png)



![png](student2_files/student2_97_5.png)



![png](student2_files/student2_97_6.png)



![png](student2_files/student2_97_7.png)



![png](student2_files/student2_97_8.png)



![png](student2_files/student2_97_9.png)



![png](student2_files/student2_97_10.png)



![png](student2_files/student2_97_11.png)



![png](student2_files/student2_97_12.png)



![png](student2_files/student2_97_13.png)



![png](student2_files/student2_97_14.png)



![png](student2_files/student2_97_15.png)



![png](student2_files/student2_97_16.png)



![png](student2_files/student2_97_17.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.000353</td>
      <td>0.863529</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.218359</td>
      <td>1.606698</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242070</td>
      <td>2.722932</td>
      <td>[basement]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.248811</td>
      <td>2.444680</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.256046</td>
      <td>2.399513</td>
      <td>[viewed]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.262110</td>
      <td>2.297609</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.314412</td>
      <td>1.462661</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.316001</td>
      <td>1.407093</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.317425</td>
      <td>1.377129</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.317450</td>
      <td>1.410740</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.317348</td>
      <td>1.424557</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.331993</td>
      <td>1.410581</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.338287</td>
      <td>1.400052</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.338279</td>
      <td>1.403466</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.342141</td>
      <td>1.367126</td>
      <td>[sqft_basement]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.353141</td>
      <td>1.707904</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.353192</td>
      <td>1.696726</td>
      <td>[yr_renovated]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> Immediately we can see that this did not work. While the % difference in MSE is low, and lower than our original model, the R-squared value was nearly cut in half. Maybe a different subset will work. We will try both above and below the 50% mark of 440,000.


```python
df_s2 = df[df.price <= 440000]
```


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_s2.drop(['price', 'price_log'], axis=1)
y = df_s2.price_log
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_s2 = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_s2.drop(['price', 'price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors")
    ax.legend()
    i += 1
results_df_s2['# Predictors'] = predictors
results_df_s2['R-squared'] = reg_score
results_df_s2['MSE % Difference'] = mse_diffs
results_df_s2['Added Predictor'] = added_pred
plt.show()
results_df_s2
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_100_1.png)



![png](student2_files/student2_100_2.png)



![png](student2_files/student2_100_3.png)



![png](student2_files/student2_100_4.png)



![png](student2_files/student2_100_5.png)



![png](student2_files/student2_100_6.png)



![png](student2_files/student2_100_7.png)



![png](student2_files/student2_100_8.png)



![png](student2_files/student2_100_9.png)



![png](student2_files/student2_100_10.png)



![png](student2_files/student2_100_11.png)



![png](student2_files/student2_100_12.png)



![png](student2_files/student2_100_13.png)



![png](student2_files/student2_100_14.png)



![png](student2_files/student2_100_15.png)



![png](student2_files/student2_100_16.png)



![png](student2_files/student2_100_17.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.076001</td>
      <td>-4.768876</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.127246</td>
      <td>-4.065290</td>
      <td>[viewed]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.156065</td>
      <td>-4.270255</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.181760</td>
      <td>-4.309963</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.189031</td>
      <td>-4.140620</td>
      <td>[basement]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.195861</td>
      <td>-4.287810</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.200470</td>
      <td>-4.028106</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.205445</td>
      <td>-3.820207</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.207971</td>
      <td>-3.825068</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.208830</td>
      <td>-3.849410</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.210833</td>
      <td>-3.541979</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.214030</td>
      <td>-3.095786</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.216662</td>
      <td>-2.789553</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.216640</td>
      <td>-2.783334</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.216492</td>
      <td>-2.533299</td>
      <td>[sqft_basement]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.223238</td>
      <td>-2.234546</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.222895</td>
      <td>-2.197282</td>
      <td>[yr_renovated]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_s3 = df[df.price >= 440000]
```


```python
#RFE regression of all predictors using repeated k-fold 

#setting X, y and creating train/test split
X = df_s3.drop(['price', 'price_log'], axis=1)
y = df_s3.price_log
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)

#initializing counter variable 'i' and creating lists to add to over each iteration
i = 1
results_df_s3 = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []

#this loop repeats once for each column in 'df'
while i <= (int(len(df_s3.drop(['price', 'price_log'], axis=1).columns))):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=47)
    scores = 0
    percent_diffs = 0
    
    #this loop using a repeated k-fold to generate an average R-squared and % difference in train/test MSE 
    for train_index, test_index in rkf.split(X_train):
        X_train2, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select = i)
        selector = selector.fit(X_train2, y_train2) 
        selected_columns = X_train2.columns[selector.support_ ]
        linreg.fit(X_train2[selected_columns],y_train2)
        scores = scores + linreg.score(X_test[selected_columns], y_test)
        y_hat_train = linreg.predict(X_train[selected_columns])
        y_hat_test = linreg.predict(X_test[selected_columns])
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        mse_diff = test_mse - train_mse
        percent_diff = mse_diff/train_mse
        percent_diffs += percent_diff
        
    #adding data to lists
    predictors.append(i)
    reg_score.append(scores/30)
    mse_diffs.append(percent_diffs/30*100)
    added_pred.append(list(set(selected_columns) - set(previous_columns)))
    previous_columns = selected_columns
    predicted = linreg.predict(X_test[selected_columns])

    fig, ax = plt.subplots(figsize=(5,5))
    #ax.scatter(y_test, predicted, edgecolors=(0, 0, 0))
    ax.plot(y_test, y_test, 'k--', lw=4, color='y')
    sns.regplot(x=y_test, y=predicted, ax=ax, line_kws={"color": "red"})
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(str(i) + " Predictors") 
    ax.legend()
    i += 1
results_df_s3['# Predictors'] = predictors
results_df_s3['R-squared'] = reg_score
results_df_s3['MSE % Difference'] = mse_diffs
results_df_s3['Added Predictor'] = added_pred
plt.show()
results_df_s3
```

    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\sparr\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](student2_files/student2_102_1.png)



![png](student2_files/student2_102_2.png)



![png](student2_files/student2_102_3.png)



![png](student2_files/student2_102_4.png)



![png](student2_files/student2_102_5.png)



![png](student2_files/student2_102_6.png)



![png](student2_files/student2_102_7.png)



![png](student2_files/student2_102_8.png)



![png](student2_files/student2_102_9.png)



![png](student2_files/student2_102_10.png)



![png](student2_files/student2_102_11.png)



![png](student2_files/student2_102_12.png)



![png](student2_files/student2_102_13.png)



![png](student2_files/student2_102_14.png)



![png](student2_files/student2_102_15.png)



![png](student2_files/student2_102_16.png)



![png](student2_files/student2_102_17.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Predictors</th>
      <th>R-squared</th>
      <th>MSE % Difference</th>
      <th>Added Predictor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.046828</td>
      <td>-2.856566</td>
      <td>[waterfront]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.326432</td>
      <td>-3.298947</td>
      <td>[grade]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.365909</td>
      <td>-3.383069</td>
      <td>[view]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.371022</td>
      <td>-2.879090</td>
      <td>[bathrooms]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.440569</td>
      <td>-3.596222</td>
      <td>[age]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.454360</td>
      <td>-3.789706</td>
      <td>[condition]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.457782</td>
      <td>-3.758151</td>
      <td>[sqft_lot15]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.464461</td>
      <td>-3.940624</td>
      <td>[basement]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.474467</td>
      <td>-4.252127</td>
      <td>[sqft_lot]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.495179</td>
      <td>-5.125408</td>
      <td>[sqft_above]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.505169</td>
      <td>-5.518740</td>
      <td>[viewed]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.513762</td>
      <td>-5.810980</td>
      <td>[bedrooms]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.516166</td>
      <td>-5.945644</td>
      <td>[sqft_living]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.516958</td>
      <td>-5.955271</td>
      <td>[floors]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.518272</td>
      <td>-5.587980</td>
      <td>[sqft_basement]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.526553</td>
      <td>-5.403242</td>
      <td>[sqft_living15]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.530526</td>
      <td>-5.923348</td>
      <td>[yr_renovated]</td>
    </tr>
  </tbody>
</table>
</div>



<h6> Although choosing a subset where 'price' is greater than or equal to 615,000 gave the best result out of these experiments. All of them perform worse than our original model and signify that it is best we include the full range of home prices. 

## Interpreting  the Data
<h4> In this final section we will use our model to draw conclusions about the data and its real-world applications.

<h6> We will look at the coefficients of our regression model. Since our dependent variable 'price' was log transformed, we will reverse that transformation for each of the coefficients in order to be able to talk about them in terms of real-life values. We will also sort the coefficients in descending order to rank the significance of each predictor.


```python
#creates a dataframe of both log transformed and non-log transformed coefficients of predictors in model
X = df.drop(['price', 'price_log'], axis=1)
y = df.price_log
model_reg = LinearRegression()
linreg.fit(X, y)
model_coef = pd.DataFrame()
preds = list()
coef_log = list()
coef = list()
sum_coef = 0
percent_coef = list()
i = 0
for col in X.columns:
    preds.append(X.columns.values[i])
    coef_log.append(linreg.coef_[i])
    coef.append(10**(linreg.coef_[i]))
    sum_coef += 10**(linreg.coef_[i])
    i += 1
model_coef['Predictor'] = preds
model_coef['Coefficient(log)'] = coef_log
model_coef['Coefficient'] = coef
for c in model_coef.Coefficient:
    percent_coef.append(c/sum_coef*100)
model_coef['Contribution(%)'] = percent_coef
model_coef.sort_values(by=['Coefficient'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predictor</th>
      <th>Coefficient(log)</th>
      <th>Coefficient</th>
      <th>Contribution(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>waterfront</td>
      <td>0.400170</td>
      <td>2.512872</td>
      <td>12.661660</td>
    </tr>
    <tr>
      <th>8</th>
      <td>grade</td>
      <td>0.192811</td>
      <td>1.558873</td>
      <td>7.854728</td>
    </tr>
    <tr>
      <th>14</th>
      <td>basement</td>
      <td>0.082538</td>
      <td>1.209310</td>
      <td>6.093378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>floors</td>
      <td>0.078084</td>
      <td>1.196973</td>
      <td>6.031212</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bathrooms</td>
      <td>0.067054</td>
      <td>1.166954</td>
      <td>5.879954</td>
    </tr>
    <tr>
      <th>15</th>
      <td>age</td>
      <td>0.064526</td>
      <td>1.160182</td>
      <td>5.845833</td>
    </tr>
    <tr>
      <th>7</th>
      <td>condition</td>
      <td>0.050389</td>
      <td>1.123023</td>
      <td>5.658600</td>
    </tr>
    <tr>
      <th>16</th>
      <td>viewed</td>
      <td>0.043601</td>
      <td>1.105606</td>
      <td>5.570842</td>
    </tr>
    <tr>
      <th>6</th>
      <td>view</td>
      <td>0.029136</td>
      <td>1.069390</td>
      <td>5.388360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sqft_living</td>
      <td>0.012926</td>
      <td>1.030210</td>
      <td>5.190940</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sqft_above</td>
      <td>0.004070</td>
      <td>1.009416</td>
      <td>5.086166</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sqft_living15</td>
      <td>0.000150</td>
      <td>1.000345</td>
      <td>5.040462</td>
    </tr>
    <tr>
      <th>11</th>
      <td>yr_renovated</td>
      <td>0.000032</td>
      <td>1.000074</td>
      <td>5.039095</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sqft_basement</td>
      <td>0.000011</td>
      <td>1.000026</td>
      <td>5.038851</td>
    </tr>
    <tr>
      <th>0</th>
      <td>bedrooms</td>
      <td>-0.034403</td>
      <td>0.923841</td>
      <td>4.654980</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sqft_lot</td>
      <td>-0.041172</td>
      <td>0.909553</td>
      <td>4.582984</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sqft_lot15</td>
      <td>-0.060652</td>
      <td>0.869656</td>
      <td>4.381955</td>
    </tr>
  </tbody>
</table>
</div>



<h6> From the above table we can see the relative impact of each predictor with 'waterfront' being the most significant and 'sqft_lot15' being the least signficant. This tells us that a home having a waterfront view has the greatest impact on the price of the home while the square footage of the lot a home sits on has the least impact on the home price. The grade of the home, whether or not the home has a basement, and the number of floors are the next most impactful factors. It is interesting that bedrooms is so low on the list as one would usually think that homes with more bedrooms cost more. This is likely because the number of bedrooms can be arbitrary in a home. Two homes with the same square footage could have differing numbers of bedrooms and perhaps having larger but smaller rooms is more appealing to home buyers. 

<h6> Running one final test on our model, using an OLS regression this time, we can check the p-values of each of our variables to make sure none of them are higher than 0.05 which would invalidate them being in our model


```python
outcome = 'price'
predictors = df.drop(['price', 'price_log'], axis=1)
pred_sum = "+".join(predictors.columns)
formula = outcome + "~" + pred_sum
model = ols(formula=formula, data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.598</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.598</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1800.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 30 Oct 2018</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:59:48</td>     <th>  Log-Likelihood:    </th> <td>-2.7750e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20587</td>      <th>  AIC:               </th>  <td>5.550e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20569</td>      <th>  BIC:               </th>  <td>5.552e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    17</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td> -7.23e+05</td> <td> 2.26e+04</td> <td>  -32.042</td> <td> 0.000</td> <td>-7.67e+05</td> <td>-6.79e+05</td>
</tr>
<tr>
  <th>bedrooms</th>      <td>-1.821e+04</td> <td> 1847.657</td> <td>   -9.857</td> <td> 0.000</td> <td>-2.18e+04</td> <td>-1.46e+04</td>
</tr>
<tr>
  <th>bathrooms</th>     <td> 1.425e+04</td> <td> 5555.921</td> <td>    2.566</td> <td> 0.010</td> <td> 3364.286</td> <td> 2.51e+04</td>
</tr>
<tr>
  <th>sqft_living</th>   <td>-8005.6144</td> <td> 1370.702</td> <td>   -5.841</td> <td> 0.000</td> <td>-1.07e+04</td> <td>-5318.930</td>
</tr>
<tr>
  <th>sqft_lot</th>      <td>-2.585e+04</td> <td> 3830.851</td> <td>   -6.748</td> <td> 0.000</td> <td>-3.34e+04</td> <td>-1.83e+04</td>
</tr>
<tr>
  <th>floors</th>        <td>  2.93e+04</td> <td> 3585.926</td> <td>    8.172</td> <td> 0.000</td> <td> 2.23e+04</td> <td> 3.63e+04</td>
</tr>
<tr>
  <th>waterfront</th>    <td> 3.985e+05</td> <td> 1.89e+04</td> <td>   21.080</td> <td> 0.000</td> <td> 3.61e+05</td> <td> 4.36e+05</td>
</tr>
<tr>
  <th>view</th>          <td>  5.34e+04</td> <td> 4928.857</td> <td>   10.834</td> <td> 0.000</td> <td> 4.37e+04</td> <td> 6.31e+04</td>
</tr>
<tr>
  <th>condition</th>     <td> 2.708e+04</td> <td> 2062.370</td> <td>   13.130</td> <td> 0.000</td> <td>  2.3e+04</td> <td> 3.11e+04</td>
</tr>
<tr>
  <th>grade</th>         <td>  1.11e+05</td> <td> 1903.138</td> <td>   58.301</td> <td> 0.000</td> <td> 1.07e+05</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>sqft_above</th>    <td> 1.646e+04</td> <td> 1316.574</td> <td>   12.503</td> <td> 0.000</td> <td> 1.39e+04</td> <td>  1.9e+04</td>
</tr>
<tr>
  <th>sqft_basement</th> <td>  174.4103</td> <td>   14.822</td> <td>   11.767</td> <td> 0.000</td> <td>  145.357</td> <td>  203.464</td>
</tr>
<tr>
  <th>yr_renovated</th>  <td>   27.2543</td> <td>    3.543</td> <td>    7.692</td> <td> 0.000</td> <td>   20.309</td> <td>   34.200</td>
</tr>
<tr>
  <th>sqft_living15</th> <td>   85.9531</td> <td>    3.249</td> <td>   26.453</td> <td> 0.000</td> <td>   79.584</td> <td>   92.322</td>
</tr>
<tr>
  <th>sqft_lot15</th>    <td>-2.839e+04</td> <td> 4034.078</td> <td>   -7.037</td> <td> 0.000</td> <td>-3.63e+04</td> <td>-2.05e+04</td>
</tr>
<tr>
  <th>basement</th>      <td> 3.005e+04</td> <td> 4778.213</td> <td>    6.288</td> <td> 0.000</td> <td> 2.07e+04</td> <td> 3.94e+04</td>
</tr>
<tr>
  <th>age</th>           <td> 3.666e+04</td> <td>  773.458</td> <td>   47.398</td> <td> 0.000</td> <td> 3.51e+04</td> <td> 3.82e+04</td>
</tr>
<tr>
  <th>viewed</th>        <td> -3.32e+04</td> <td> 1.19e+04</td> <td>   -2.795</td> <td> 0.005</td> <td>-5.65e+04</td> <td>-9917.970</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>7818.700</td> <th>  Durbin-Watson:     </th> <td>   1.954</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>69347.959</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.580</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>11.418</td>  <th>  Cond. No.          </th> <td>3.86e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.86e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



<h6> From our OLS regression we can see that the p-value is less than the threshold of 0.05 for each of our variables, thus allowing us to reject the null hypothesis and keep all of the predictors in our model as they are highly like to be significant. 

## Conclusion
<h4> In this project we followed the OSEMiN framework in order to build a predictive model about home sale prices in King County, Washington. Through scrubbing and transforming the data, we built a moderately well fitted model and used that model to view the contribution of each predictor to the price of a home. Our model was not a perfect fit and further investigation using different transformations and more advanced regression methods would likely produce a more robust, well-fitted model. The model could also be improved by having more data on higher priced homes as the dataset having many outliers likely worsened the fit of the regression.  

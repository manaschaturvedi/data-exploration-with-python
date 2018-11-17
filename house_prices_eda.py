import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('train.csv')

# print(df_train.columns)
# print(df_train['SalePrice'].describe())

# sns.distplot(df_train['SalePrice'])
# plt.show()

"""
Observations from the distribution chart:
- data deviates from the normal distribution
- has appreciable positive skewness (long tail is on the positive side of the peak)
- shows peakedness
"""

## skewness
# print("Skewness: %f" % df_train['SalePrice'].skew())

"""
If the skewness is between -0.5 and 0.5, the data is fairly symmetrical
If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data is moderately skewed
If the skewness is less than -1 or greater than 1, the data is highly skewed
"""

"""
Initial set of features to consider based on intuition:
- OverallQual
- YearBuilt.
- TotalBsmtSF.
- GrLivArea.
"""

## RELATIONSHIP WITH NUMERICAL VARIABLES

## scatter plot grlivarea/saleprice: linear relationship
# data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
# plt.show()

## scatter plot totalbsmtsf/saleprice
# data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
# data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
# plt.show()

## correlation between two variables (in a linear relationship, correlation is 1)
# print(df_train['GrLivArea'].corr(df_train['SalePrice']))
# print(df_train['TotalBsmtSF'].corr(df_train['SalePrice']))

## RELATIONSHIP WITH CATEGORICAL VARIABLES

## box plot overallqual/saleprice
# data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.show()

## box plot YearBuilt/saleprice
# data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
# f, ax = plt.subplots(figsize=(16, 8))
# fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)
# plt.show()

"""
Note: we don't know if 'SalePrice' is in constant prices. Constant prices try to remove the 
effect of inflation. If 'SalePrice' is not in constant prices, it should be, so than prices 
are comparable over the years.

- 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships 
are positive, which means that as one variable increases, the other also increases. In the case 
of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.

- 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems
to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase
with the overall quality.
"""

"""
Until now we just followed our intuition and analysed the variables we thought were important. In spite of our 
efforts to give an objective character to our analysis, we must say that our starting point was subjective.
"""

## correlation matrix
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

"""
At first sight, there are two white colored squares that get my attention. The first one refers to the 'TotalBsmtSF' and
'1stFlrSF' variables, and the second one refers to the 'GarageX' variables. Both cases show how significant the
correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of 
multicollinearity.
Also check out the correlations of other variables with 'SalePrice'. Observations:
- 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
- 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, the number of cars that 
    fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. You'll 
    never be able to distinguish them. Therefore, we just need one of these variables in our analysis 
    (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
- 'TotalBsmtSF' and '1stFloor' also seem to be twin brothers. We can keep 'TotalBsmtSF' just to say that our first guess
     was right
- Honorable mentions: 'FullBath', 'YearBuilt
"""

## HANDLING MISSING DATA
"""
Important questions when thinking about missing data:

- How prevalent is the missing data?
- Is missing data random or does it have a pattern?

The answer to these questions is important for practical reasons because missing data can imply a reduction of the 
sample size. This can prevent us from proceeding with the analysis. Moreover, from a substantive perspective, we need 
to ensure that the missing data process is not biased and hiding an inconvenient truth.
"""

## missing data
# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(20))

"""
Let's analyse this to understand how to handle the missing data.

We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it
never existed. This means that we will not try any trick to fill the missing data in these cases. According to this,
there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that we should delete. The point is: will we 
miss this data? I don't think so. None of these variables seem to be very important, since most of them are not aspects 
in which we think about when buying a house (maybe that's the reason why data is missing?). Moreover, looking closer at 
the variables, we could say that variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates for 
outliers, so we'll be happy to delete them.

In what concerns the remaining cases, we can see that 'GarageX' variables have the same number of missing data.
Since the most important information regarding garages is expressed by 'GarageCars' and considering that we are just 
talking about 5% of missing data, I'll delete the mentioned 'GarageX' variables. The same logic applies to 'BsmtX' 
variables.

Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have 
a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, we will not lose information 
if we delete 'MasVnrArea' and 'MasVnrType'.

Finally, we have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation 
and keep the variable.

In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'.
In 'Electrical' we'll just delete the observation with missing data.
"""

## dealing with missing data
# df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
# df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# print(df_train.isnull().sum().max()) # just checking that there's no missing data remaining
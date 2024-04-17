# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/96eef140-fc5e-4963-bcbc-6cea4e6b3dd1)
```
data.isnull().sum()
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/2eca15bd-c81a-4f5e-a558-f039c2390dc5)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/aaad90f5-32af-46f7-a889-bc8d71fdfd5a)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/12167e86-63a5-4aa6-8610-e076357adb56)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/559615e6-6ab0-43b5-ad14-7898a220f373)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/1d79f56f-716a-485f-b6c4-d9eaa02d148e)
```
data2
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/f067da18-cc05-4c82-b9f7-159cdc775a57)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/43f2f5f6-7980-4a45-9394-e233c16c117a)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/87db6aa8-c280-4935-8e0d-885822906fdd)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/4dae6057-2da7-4482-90cb-cbaadc75fbb8)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/e8c287e7-8bc1-4dd0-9b6b-1645d058f4f4)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/3af5bd3d-293a-4957-a87d-3ba81b59f1f0)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/a4bc133a-548b-4c76-8d9a-8f82f222ec91)
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/a92b443c-be05-4c5a-80a5-f0b57dfbdf9a)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/d408ec66-c621-4e21-a327-fa53a55e4568)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/ad6e5e38-c562-4895-9469-0a11f4e794c0)
```
data.shape
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/28c4e531-f7d5-46a2-bd7b-e0c1af45dabf)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/6b3919ec-9592-4990-b7fa-217568ac84df)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/6a0e8b22-040b-459f-bebc-18e782a615f0)
```
tips.time.unique()
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/33f26fa9-a6a8-44b4-8c23-c4a32f45071d)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/58a883c5-197d-404e-9da8-96c8c3752523)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Vanisha0609/EXNO-4-DS/assets/119104009/d3f251ab-7a5b-4109-b1dd-e7b3ac317cfd)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.

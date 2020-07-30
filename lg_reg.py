from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

#! here we load the same credit approval dataset used for XGBClassifier
#! Logistic regression gives us an accuracy of 83.09%
#! XGBClassifier gave an accuracy of 85.02%

df1 = pd.read_csv('crx.data', delimiter=',')
df1.columns = [i for i in range(1, 17)]
# print(df1.head)

# * split dataset
X = df1.iloc[:, :15]
Y = df1.iloc[:, 15]

seed = 4
test_size = 0.3
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=test_size, random_state=seed)

# * instantiate label encoder and standard scaler
lb = LabelBinarizer()
le = LabelEncoder()
ss = StandardScaler()
# * get colun names that have objects and mask them to a list
mask = Xtrain.dtypes == object
categorical_cols = Xtrain.columns[mask].tolist()

# * apply label encoder to the columns masked earlier
Xtrain[categorical_cols] = Xtrain[categorical_cols].apply(
    lambda col: le.fit_transform(col))
Xtest[categorical_cols] = Xtest[categorical_cols].apply(
    lambda col: le.fit_transform(col))
# print(Xtrain.head)

Xtrain = pd.DataFrame(ss.fit_transform(
    Xtrain), index=Xtrain.index, columns=Xtrain.columns)
Xtest = pd.DataFrame(ss.fit_transform(
    Xtest), index=Xtest.index, columns=Xtest.columns)

Ytrain = pd.DataFrame(lb.fit_transform(
    Ytrain), index=Ytrain.index)

model = LogisticRegression()
model.fit(Xtrain, Ytrain.values.ravel())

ypred = model.predict(Xtest)
ypred = lb.inverse_transform(ypred)
accuracy = accuracy_score(Ytest, ypred)
print(f'accuracy = {accuracy*100}%')

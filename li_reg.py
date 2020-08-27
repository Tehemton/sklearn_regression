import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
import os
import pprint
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pickle

#! this program will develop an linear reqression model
#! compare RMSE for Liner regression with XGBoost
#! LR has  an RMSE of 14.815 on the emissions dataset
#! XGB has an RMSE of 12.31

# * load the data
li = []
for file in os.listdir('.\\emission_data'):
    imp = pd.read_csv(f'.\\emission_data\\{file}', index_col=None, header=0)
    li.append(imp)

df1 = pd.concat(li, axis=0, ignore_index=True)
# print(df1.head)


df1.plot(subplots=True, layout=(6, 2))
plt.show()
# print(df1.head)

# * split into X and Y
X = df1.iloc[:, 0:10]
Y = df1.iloc[:, 10]
# print(X.shape, Y.shape)

# * split sets
split = 0.6

Xtrain, Xtest = X.iloc[:int(len(X.index)*split),
                       :], X.iloc[int(len(X.index)*split):, :]
# print(Xtrain.shape, '\t', Xtest.shape)
Ytrain, Ytest = Y.iloc[:int(len(Y.index)*split)
                       ], Y.iloc[int(len(Y.index)*split):]

#! converting target to numpy array and rehaping since scalerneeds it that way
Ytrain = Ytrain.to_numpy()
Ytrain = Ytrain.reshape((-1, 1))

rbs = RobustScaler()
rbs.fit(Xtrain)

rbsy = RobustScaler()
rbsy.fit(Ytrain)

Xtrain = pd.DataFrame(rbs.transform(
    Xtrain), index=Xtrain.index, columns=Xtrain.columns)
Xtest = pd.DataFrame(rbs.transform(
    Xtest), index=Xtest.index, columns=Xtest.columns)
Ytrain = pd.DataFrame(rbsy.transform(
    Ytrain))  # , index=Ytrain.index, columns=Ytrain.columns)


if not os.path.isfile('.\\model.dat'):
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    pickle.dump(model, open("model.dat", "wb"))

else:
    model = pickle.load(open("model.dat", "rb"))

ypred = model.predict(Xtest)
ypred = ypred.reshape((-1, 1))

ypred = rbsy.inverse_transform(ypred)
rmse = sqrt(mean_squared_error(Ytest, ypred))
print(rmse)

Ytest_np = Ytest.to_numpy()
ypred = ypred.flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(y=Ytest_np,
                         mode='lines',
                         name='Ytest'))
fig.add_trace(go.Scatter(y=ypred,
                         mode='lines',
                         name='ypred'))
fig.write_html(f'.\\Ytest.html')

new = 2
chromepath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chromepath).open(f'.\\Ytest.html', new=new)

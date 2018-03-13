# -*- coding: utf-8 -*-
"""

@author: Samraj
 """

import pandas as pd
import numpy as np

readData = pd.read_csv('E:/DA/Thesis/VideoGamesCS.csv'
                       ,engine = 'python')
readData.head()
#remove the nan data if developer row is nan
readData= readData.dropna(subset = ['Platform','Publisher','Genre'
                                           ,'Developer','Rating'],how='any')

"""
Assigning Predictor and prediction Varibale to Xtrain and Ytrain

"""
Ytrain = readData['Global_Sales']
Xtrain = readData.drop(labels = ['Name','Year_of_Release','Global_Sales'
                                 ,'User_Count'
                                 ,'Critic_Count'], axis = 1)


""" Grouping Critic Score into three categories """
def classifyScore(x):
    if x > 80:
        return 'High'
    if 40 < x <80:
        return 'Moderate'
    if x < 40:
        return 'Low'

""" Grouping User Score into three categories """
def classifyScore1(x):
    if x > 8:
        return 'High'
    if 4 < x <8:
        return 'Moderate'
    if x < 4:
        return 'Low'
        
Xtrain['Critic_Score'] = Xtrain['Critic_Score'].apply(lambda x: classifyScore(x))
Xtrain['User_Score'] = Xtrain['User_Score'].apply(lambda x: classifyScore1(x))

""" Fitting the Train Data """

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
Xtrain['Platform'] = le.fit_transform(Xtrain['Platform'])
Xtrain['Publisher'] = le.fit_transform(Xtrain['Publisher'].astype(str))
Xtrain['Genre'] = le.fit_transform(Xtrain['Genre'])
Xtrain['Developer'] = le.fit_transform(Xtrain['Developer'])
Xtrain['Rating'] = le.fit_transform(Xtrain['Rating'].astype(str))
Xtrain['User_Score'] = le.fit_transform(Xtrain['User_Score'].astype(str))
Xtrain['Critic_Score'] = le.fit_transform(Xtrain['Critic_Score'].astype(str))

""" Encoding and Scaling for stability """
ohe = OneHotEncoder(categorical_features = [0,1,2,7])
Xtrain = ohe.fit_transform(Xtrain).toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)

""" Splitting Data for Train and test """
from sklearn.model_selection import train_test_split
Xtrain,x_test,Ytrain,y_test = train_test_split(Xtrain,Ytrain
                                                 ,test_size = 0.5
                                                 ,random_state = 42)


from keras.layers import Dense
from keras.models import Sequential

""" Creating Hidden layer with multiple avtivations to perform Deep Learning """
predModel = Sequential()
predModel.add(Dense(units = 2048,activation= 'linear',kernel_initializer='uniform'
                     ,input_dim = 303))
predModel.add(Dense(units = 1024, activation = 'linear',kernel_initializer='uniform'))
predModel.add(Dense(units = 512, activation = 'sigmoid',kernel_initializer='uniform'))
predModel.add(Dense(units = 128, activation = 'sigmoid',kernel_initializer='uniform'))
predModel.add(Dense(units = 64, activation = 'sigmoid',kernel_initializer='uniform'))
predModel.add(Dense(units = 32, activation = 'sigmoid',kernel_initializer='uniform'))
predModel.add(Dense(units = 1,activation = 'linear',kernel_initializer='uniform'))
predModel.compile(optimizer = 'adam',loss = 'mse', metrics=['accuracy'])

predModel.fit(Xtrain,Ytrain,batch_size = 32,epochs = 30
               ,validation_data = (x_test,y_test))

history = predModel.fit(Xtrain,Ytrain,batch_size = 32,epochs = 30
               ,validation_data = (x_test,y_test))
score = predModel.evaluate(x_test, y_test, batch_size=32)
# Predicting the Test set results
y_pred = predModel.predict(Xtrain)
# calculate the accuracy of our model
acc = [sum(y_pred)/sum(np.float32(Ytrain))]
print(" Accuracy ",acc)
#see hows our model works
import matplotlib.pyplot as plt
plt.plot(np.float32(Ytrain), color ='red', label = 'Actual Sales')
plt.plot(y_pred, color = 'blue', label = 'Predicted Sales')
plt.title('Video Games Sales Prediction Graph')
plt.ylabel('Video game Global Sale')
plt.legend
plt.show()

from sklearn.metrics import explained_variance_score
variance = explained_variance_score(Ytrain, y_pred)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
mae = mean_absolute_error(Ytrain, y_pred)
meanLog = mean_squared_log_error(Ytrain, y_pred) 
medianError = median_absolute_error(Ytrain, y_pred)
R2 =r2_score(Ytrain, y_pred) 
mse = mean_squared_error(Ytrain, y_pred)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
print('variance:', variance)
print('MAE:', mae)
print('meanLog:', meanLog)
print('medianError:', medianError)
print('R2: ',R2)


""" Plotting Curve for Accuracy and Loss """ 

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
""" Plotting Curve for Varaible loss and Loss """ 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

fig, ax = plt.subplots()
ax.scatter(Ytrain, y_pred, edgecolors=(0, 0, 0))
ax.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
# -*- coding: utf-8 -*-
"""
@author: Samraj
"""

import pandas as pd
import numpy as np
import defFile as lib
import matplotlib.pyplot as plt
import seaborn as sns

readData = pd.read_csv('E:/DA/Thesis/VideoGamesCS.csv', encoding="utf-8")

#Visualizing Columns which are required to predict the HIT
Modcols = ['Publisher','Developer', 'Genre','Platform','Rating']
#countVisualization = lib.drawCountGraph(Modcols,readData)

#PolyNomial Data to Integer Classification Automatically
#Columns which requires Conversion
convCols = ['Publisher','Developer', 'Genre','Platform','Rating']

#Storing Converted Data
convData = lib.polynomialTOInteger(convCols,readData)

#Separating Data which Needs to be Indexed
indexData = convData[['Name','Platform','Genre','Publisher','Year_of_Release','Critic_Score','User_Score','Global_Sales']]
indexData = indexData.dropna().reset_index(drop=True)

#Pulling Numerical Data to Train Our Model
finalData = indexData[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales','User_Score']]

# Global Sales as Prediction Variable
finalData['Global_Sales_Classified'] = finalData['Global_Sales']
finalData.drop('Global_Sales', axis=1, inplace=True)

#Model Classification - Whether Global Sale is A SUCCESS or NOT
#Classifying Data INTO 1 and 0
#1 for Sales More than 1 Million, NOTE: DATASET Sales in Million Units
def hitClassify(globalSales):
    if globalSales >= 1:
        return 1
    else:
        return 0

finalData['Global_Sales_Classified'] = finalData['Global_Sales_Classified'].apply(lambda x: hitClassify(x))

scoreLvl = readData.dropna(subset=['Critic_Score']).reset_index(drop=True)
scoreLvl['Score_Group'] = scoreLvl['Critic_Score'].apply(lambda x: lib.criticScoreCategory(x))

#Columns To be Plotted
catCol = ['Publisher', 'Developer','Genre', 'Platform']
scoreGroupGraph = lib.drawCountGraph2(catCol,scoreLvl)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
def calMetric(ytest,y_pred_train,model):
    confusion = confusion_matrix(ytest, y_pred_train)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print(" Model : ", model)
    # use float to perform true division, not integer division
    print((TP + TN) / float(TP + TN + FP + FN))
    print("Accuracy ",metrics.accuracy_score(ytest, y_pred_train))
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    
    print("Classification Error ",classification_error)
    print("CE Metric",1 - metrics.accuracy_score(ytest, y_pred_train))
    sensitivity = TP / float(FN + TP)
    
    print("Sensitivity: ",sensitivity)
    print(" Recall ",metrics.recall_score(ytest, y_pred_train))
    specificity = TN / (TN + FP)
    
    print(" Specificity",specificity)
    false_positive_rate = FP / float(TN + FP)
    
    print(" False Positive Rate: ",false_positive_rate)
    print(" 1- Specificity",1 - specificity)
    precision = TP / float(TP + FP)
    
    print(" Precision ",precision)
    print(" Precision Metric: ",metrics.precision_score(ytest, y_pred_train))
    

def mse(pred, actual):
    return ((pred - actual)**2).mean()

# Creating a copy of Final Data for This Regression Model
regData = finalData

#Setting Global Sales Classified Data as Y
y = regData['Global_Sales_Classified'].values
          
#Removing Prediction Classifier to Train the Support Varibles
regData = regData.drop(['Global_Sales_Classified'],axis=1)

X = regData.values

#Predicting Hit of Video Game Sale Globally

#Training the Dataset for Random Forest
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50, random_state=2)
print("****************************************************************")
ranDam = RandomForestClassifier(random_state=2).fit(Xtrain, ytrain)
yPredicted = ranDam.predict_proba(Xtest)
print("Random Forest Accuracy: ", sum(pd.DataFrame(yPredicted).idxmax(axis=1).values
                                   == ytest)/len(ytest))

all_pred = ranDam.predict(Xtest)
print(classification_report(ytest, all_pred))
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(ytest, all_pred), annot=True, linewidths=.5, ax=ax, fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Training Set Confusion Matrix')
plt.show()
print(" MSE RF ",mse(all_pred, ytest))
calMetric(ytest,all_pred,'Random Forest')
fpr, tpr, thresholds = metrics.roc_curve(ytest, all_pred)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Random Forest Classification ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

print("****************************************************************")
# Logestic Prediction
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.70)
log_reg = LogisticRegression().fit(Xtrain, ytrain)
y_val_2 = log_reg.predict_proba(Xtest)
print("Log Regression Accuracy: ", sum(pd.DataFrame(y_val_2).idxmax(axis=1).values
                                    == ytest)/len(ytest))
all_predictions = log_reg.predict(Xtest)
print(classification_report(ytest, all_predictions))
fig, ax = plt.subplots(figsize=(3.5,2.5))
sns.heatmap(confusion_matrix(ytest, all_predictions), annot=True, linewidths=.5, ax=ax, fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')
sns.plt.title('Training Set Confusion Matrix')
plt.show()
print(" MSE LR ",mse(all_predictions, ytest))
calMetric(ytest,all_predictions,'Log Regression')
fpr, tpr, thresholds  = metrics.roc_curve(ytest, all_predictions)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Logistic Regression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

print("****************************************************************")        
# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)


# train the model using X_train_dtm & y_train
nb.fit(Xtrain, ytrain)
# calculate predicted probabilities for X_test_dtm
#y_pred_prob = nb.predict(Xtrain)
# compute the accuracy of training data predictions
y_pred_train = nb.predict(Xtest)
print("Naive Bayes model Accuracy: ", sum(pd.DataFrame(y_pred_train).idxmax(axis=1).values
                                    == ytest)/len(ytest))
print(classification_report(ytest, y_pred_train))
ax = plt.subplots(figsize=(3.5,2.5))
# look at the confusion matrix for y_test
#print(confusion_matrix(ytest, y_pred_train))
print(" MSE NB ",mse(y_pred_train, ytest))
fpr, tpr, thresholds = metrics.roc_curve(ytest, y_pred_train)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Naive Bayes')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
calMetric(ytest,y_pred_train,'Naive bayes')
print("****************************************************************")

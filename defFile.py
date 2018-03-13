# -*- coding: utf-8 -*-
"""

@author: Samraj
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def drawCountGraph(cols,data):
    for col in cols:
        chart = data[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
        sns.set_style("white")
        plt.figure(figsize=(12.4, 5))
        plt.xticks(rotation=90)
        colors = sns.color_palette('gist_earth', len(cols))
        if(col == "Genre"):
            colors = sns.color_palette('CMRmap', len(cols))
        if(col == "Developer"):
            colors = sns.color_palette('Set3', len(cols))
        sns.barplot(x=col, y='Name', data=chart[:30], palette=colors).set_title((' Number of Games by '+col), fontsize=16)
        plt.ylabel(' Game Count', fontsize=14)
        plt.xlabel('')


def drawCountGraph2(cols,scoreLvl):
    def in_top(x):
        if x in pack:
            return x
        else:
            pass
       
    for col in cols:
        pack = []
        top = scoreLvl[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()[:15]
        for x in top[col]:
            pack.append(x)
        scoreLvl[col] = scoreLvl[col].apply(lambda x: in_top(x))
        scoreLvlplatform = scoreLvl[[col, 'Score_Group', 'Global_Sales']].groupby([col, 'Score_Group']).median().reset_index().pivot(col, "Score_Group", "Global_Sales")
        plt.figure(figsize=(15, 10))
        sns.heatmap(scoreLvlplatform, annot=True, fmt=".2g", linewidths=.5).set_title((' \n'+col+' vs. critic score (by median sales) \n'), fontsize=18)
        plt.ylabel('', fontsize=14)
        plt.xlabel('Score group \n', fontsize=12)
        pack = []
    
        
def polynomialTOInteger(Label,data):
    for i in Label:
        uniques = data[i].value_counts().keys()
        order = 1
        uniquesArray = {}
        for everyValue in uniques:
            uniquesArray[everyValue] = order
            order += 1
        for key, val in uniquesArray.items():
            data.loc[data[i] == key, i] = val 
    return data

def drawCrossCat(Label,data) :
    
    #Search for the categorical types
    col_cat = []
    for col in Label :
        if data[col].dtypes == object :
            col_cat.append(col)
    
    #Size the figure that will contain the subplots
    plt.figure(figsize=(15, 30))
    i = 1
    
    #For each column
    for col in col_cat :
        col_cat.remove(col)
        for col2 in col_cat :
        
            #Plot the values
            plt.subplot(int("42" + str(i)))
            table_count = pd.pivot_table(data,values=['Global_Sales'],index=[col],columns=[col2],aggfunc='count',margins=False)
            sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

            #Add information
            plt.title("{} co-plotted with {}".format(col, col2))
        
            #Rotate the x-label
            plt.xticks(rotation=90)
            i += 1
        
    #Adjust the subplots so that they don't overlap
    plt.subplots_adjust(hspace=0.5)
    
def criticScoreCategory(score):
    if score >= 90:
        return '90-100'
    elif score >= 80:
        return '80-89'
    elif score >= 70:
        return '70-79'
    elif score >= 60:
        return '60-69'
    elif score >= 50:
        return '50-59'
    else:
        return '0-49'
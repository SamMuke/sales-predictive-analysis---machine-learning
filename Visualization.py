# -*- coding: utf-8 -*-
"""
@author: Samraj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

readData = pd.read_csv('E:/DA/Thesis/VideoGamesCS.csv', encoding="utf-8")
rpy = readData['Global_Sales'] / (2017 - readData['Year_of_Release'])
ms = readData['Global_Sales'] / readData['Global_Sales'].sum()
readData['RevPG'] = rpy
readData['MktShr'] = ms

mktShrData = readData.pivot_table('MktShr', columns='Publisher', index='Global_Sales')
RevPGData = readData.pivot_table('RevPG', columns='Publisher', index='Global_Sales')

EA1 = mktShrData['Electronic Arts']
Act1 = mktShrData['Activision']
Ubi1 = mktShrData['Ubisoft']
Nint1 = mktShrData['Nintendo']
Sony1 = mktShrData['Sony Computer Entertainment']
Tktwo1 = mktShrData['Take-Two Interactive']
Namc1 = mktShrData['Namco Bandai Games']
Kona1 = mktShrData['Konami Digital Entertainment']
THQ1 = mktShrData['THQ']
Seg1 = mktShrData['Sega']

lizt1 = [EA1, Act1, Ubi1, Nint1, Sony1, Tktwo1, Namc1, Kona1, THQ1, Seg1]
mktShrData = pd.concat(lizt1, ignore_index=True, axis=1)
mktShrData.columns = ['Electronic Arts', 'Activision', 'Ubisoft', 'Nintendo', 'Sony Computer Entertainment', 'Take-Two Interactive', 'Namco Bandai Games', 'Konami Digital Entertainment', 'THQ', 'Sega']
mktShrData.index = range(0, len((mktShrData)))

EA2 = RevPGData['Electronic Arts']
Act2 = RevPGData['Activision']
Ubi2 = RevPGData['Ubisoft']
Nint2 = RevPGData['Nintendo']
Sony2 = RevPGData['Sony Computer Entertainment']
Tktwo2 = RevPGData['Take-Two Interactive']
Namc2 = RevPGData['Namco Bandai Games']
Kona2 = RevPGData['Konami Digital Entertainment']
THQ2 = RevPGData['THQ']
Seg2 = RevPGData['Sega']

lizt2 = [EA2, Act2, Ubi2, Nint2, Sony2, Tktwo2, Namc2, Kona2, THQ2, Seg2]
RevPGData = pd.concat(lizt2, ignore_index=True, axis=1)
RevPGData.columns = ['Electronic Arts', 'Activision', 'Ubisoft', 'Nintendo', 'Sony Computer Entertainment', 'Take-Two Interactive', 'Namco Bandai Games', 'Konami Digital Entertainment', 'THQ', 'Sega']
RevPGData.index = range(0, len((mktShrData)))

fig = plt.figure(figsize=(12,8))
graph = sns.swarmplot(x=np.log(mktShrData['Activision']), y=np.log(RevPGData['Activision']), label='Activision')
graph = sns.swarmplot(x=np.log(mktShrData['Ubisoft']), y=np.log(RevPGData['Ubisoft']), label='Ubisoft')
graph = sns.swarmplot(x=np.log(mktShrData['Nintendo']), y=np.log(RevPGData['Nintendo']), label='Nintendo')
graph = sns.swarmplot(x=np.log(mktShrData['Take-Two Interactive']), y=np.log(RevPGData['Take-Two Interactive']), label='Take=Two Interactive')
graph = sns.swarmplot(x=np.log(mktShrData['Sony Computer Entertainment']), y=np.log(RevPGData['Sony Computer Entertainment']), label='Sony Computer Entertainment')
graph = sns.swarmplot(x=np.log(mktShrData['Electronic Arts']), y=np.log(RevPGData['Electronic Arts']), label='Electronic Arts')
graph = sns.swarmplot(x=np.log(mktShrData['Namco Bandai Games']), y=np.log(RevPGData['Namco Bandai Games']), label='Namco Bandai Games')
graph = sns.swarmplot(x=np.log(mktShrData['Konami Digital Entertainment']), y=np.log(RevPGData['Konami Digital Entertainment']), label='Konami Digital Entertainment')
graph = sns.swarmplot(x=np.log(mktShrData['THQ']), y=np.log(RevPGData['THQ']), label='THQ')
graph = sns.swarmplot(x=np.log(mktShrData['Sega']), y=np.log(RevPGData['Sega']), label='Sega')



graph.set_xlabel(xlabel='Revenue Per Year by Games', fontsize=16)
graph.set_ylabel(ylabel='Market Share by Game', fontsize=16)
graph.set_title(label='Revenue Per Year Vs Market Share by Top 10 Publishers', fontsize=20)

plt.tick_params(axis='x', which='both', bottom='off',
                top='off', labelbottom='off')
plt.show();
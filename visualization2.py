import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

readData = pd.read_csv('E:/DA/Thesis/VideoGamesCS.csv', encoding="utf-8")

def groupCompanyByPlatform(platform):
    if(platform == 'PS3' or platform == 'PS4' or platform == 'PSP' or platform == 'PSV' or platform == 'PS2' or platform == 'PS'):
        return 'Sony'
    if(platform == 'X360' or platform == 'XOne' or platform == 'XB' ):
        return 'Microsoft'
    if(platform == 'WiiU' or platform == '3DS' or platform == 'Wii' or platform == 'PSV' or platform == 'PSV'):
        return 'Nintendo'
    if(platform == "GEN" or platform =="SCD"or platform =="DC"or platform =="GG" or platform =="SAT"):
        return 'Sega'
    if(platform == "2600" or platform =="3DO"or platform =="NG"or platform =="WS" or platform =="PCFX"):
        return 'Others'
    
readData['CompanyName'] = readData['Platform'].apply(lambda x: groupCompanyByPlatform(x))
   
comGroup = readData.groupby(['CompanyName']).sum().loc[:, 'NA_Sales':'Global_Sales']
comGroup['NA_Sales%'] = comGroup['NA_Sales']/comGroup['Global_Sales']
comGroup['EU_Sales%'] = comGroup['EU_Sales']/comGroup['Global_Sales']
comGroup['JP_Sales%'] = comGroup['JP_Sales']/comGroup['Global_Sales']
comGroup['Other_Sales%'] = comGroup['Other_Sales']/comGroup['Global_Sales']

plt.figure(figsize=(8, 10))
sns.set(font_scale=0.7)
plt.subplot(211)
sns.heatmap(comGroup.loc[:, 'NA_Sales':'Other_Sales'], annot=True, fmt = '.1f')
plt.title("Comparation of each area in each Genre")
plt.subplot(212)

sns.heatmap(comGroup.loc[:,'NA_Sales%':'Other_Sales%'], vmax =1, vmin=0, annot=True, fmt = '.2%',cmap="BuPu")
plt.title("Comparation of each area in each Genre(Pencentage)")
plt.show()

EU = readData.pivot_table('EU_Sales', columns='Name', index='Genre', aggfunc='mean').mean(axis=1)
NA = readData.pivot_table('NA_Sales', columns='Name', index='Genre', aggfunc='mean').mean(axis=1)
JP = readData.pivot_table('JP_Sales', columns='Name', index='Genre', aggfunc='mean').mean(axis=1)
Other = readData.pivot_table('Other_Sales', columns='Name', index='Genre', aggfunc='mean').mean(axis=1)
years = Other.index.astype(str)
regions = ['Europe','North America','Japan','Rest of the World']
plt.figure(figsize=(12,8))
ax = sns.pointplot(x=years, y=EU, color='red', scale=0.7)
ax = sns.pointplot(x=years, y=NA, color='blue', scale=0.7)
ax = sns.pointplot(x=years, y=JP, color='green', scale=0.7)
ax = sns.pointplot(x=years, y=Other, color='orange', scale=0.7)
ax.set_xticklabels(labels=years, fontsize=12, rotation=50)
ax.set_xlabel(xlabel='Genre of Games', fontsize=16)
ax.set_ylabel(ylabel='Mean Revenue in $ Millions', fontsize=16)
ax.set_title(label='Regional Mean Distribution of Revenue in Million ', fontsize=20)
ax.legend(handles=ax.lines[::len(years)+1], labels=regions, fontsize=18)
plt.show();

import urllib.request as urllib2
import csv as csv 
from bs4 import BeautifulSoup
import re
import time

startTime = time.time()
"""

MetaCritic Platform Names Have abbrevated or Elaborated Form
So changing Platform Name Obtained from VGChartz accordingly to MetaCritiv

"""
metacriticPlatform = {'PS3': 'playstation-3',
					   'X360': 'xbox-360',
					   'PC': 'pc',
					   'WiiU': 'wii-u',
					   '3DS': '3ds',
					   'PSV': 'playstation-vita',
					   'iOS': 'ios',
					   'Wii': 'wii',
					   'DS': 'ds',
					   'PSP': 'psp',
					   'PS2': 'playstation-2',
					   'PS': 'playstation',
					   'XB': 'xbox', # original xbox
					   'GC': 'gamecube',
					   'GBA': 'game-boy-advance',
					   'DC': 'dreamcast',
					   'PS4': 'playstation-4',
					   'XOne': 'xbox-one'
					   }
game = {}

"""

Fetching Data From Metacritic.com For the URL Generated from VGChartz Data

"""

def fetchDataFromMetaCritic(metaURL):
    vgUrl2=metaURL
    try:
        def fetchData(soup):
            TITLE=soup.find('div',{'class':'product_title'})
            game["title"] = (TITLE.find("span",{'itemprop':'name'}).text.strip())
            DevGenRat=soup.find('div',{'class':'section product_details'}).find("div", class_="details side_details")
            game["developer"] = DevGenRat.find("li", class_="summary_detail developer").find("span", class_="data").text.strip()
            #game["genre"] = DevGenRat.find("li", class_="summary_detail product_genre").find("span", class_="data").text.strip()
            game["rating"] = DevGenRat.find("li", class_="summary_detail product_rating").find("span", class_="data").text.strip()
            users = soup.find("div", class_="details side_details")
            game["userScore"] = users.find("div", class_="metascore_w").text.strip()
            game["userCount"] = users.find("span", class_="count").a.text.replace('Ratings','').lstrip()
            game["publisher"] = soup.find("li", class_="summary_detail publisher").a.text.strip()
            game["release"] = soup.find("li", class_="summary_detail release_data").find("span", class_="data").text.strip()
            year = game["release"].split(',')
            game["releaseYear"] = year[1].lstrip()
            critic = soup.find("div", class_="details main_details")
            game["criticScore"] = critic.find("span", itemprop="ratingValue").text.strip()
            game["criticCount"] = critic.find("span", itemprop="reviewCount").text.strip()
            
        req = urllib2.Request(vgUrl2)
        req.add_unredirected_header('User-Agent','Mozilla/5.0')
        metacritic_url = urllib2.urlopen(req, timeout = 10)
        soup = BeautifulSoup(metacritic_url,'html.parser')
        fetchData(soup)
    except:
        print( "WARNING: Skipping Irregular Data ")
        pass
    
"""
Platforms Available in Meta Critic Website 

Ignoring Rest of the Platforms Since Ratings not Available

"""
platforms_to_include = ['PS3', 'X360', 'WiiU', '3DS', 'PSV', 'iOS', 'Wii', 'PSP', 'PS4', 'XOne']

#Creating CSV Files for Storing Extacted Data

filename = "vgSalesUncleaned.csv"
csvFile = open(filename, "w", encoding='UTF-16')
writeGame = csv.writer(csvFile)
writeGame.writerow(['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'release date', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'])
#Name	Platform	Year_of_Release	Genre	Publisher	NA_Sales	EU_Sales	JP_Sales	Other_Sales	Global_Sales	Critic_Score	Critic_Count	User_Score	User_Count	Developer	Rating

""""
To extract Total Number of Games 

"""
vgURL = 'http://www.vgchartz.com/gamedb/?name=&publisher=&platform=&genre=&minSales=30&results='
req = urllib2.Request(vgURL)
req.add_unredirected_header('User-Agent','Mozilla/5.0')
metacritic_url = urllib2.urlopen(req, timeout=50000)
soup = BeautifulSoup(metacritic_url) 
columns = soup.findAll('td', text = re.compile('Numbers '))
totalGames = columns[0].get_text().split('of')
totalGames = int(totalGames[1].lstrip().replace(',',''))

print("Total Games Available: ", totalGames)
totalPages = totalGames/1000
totalPages = int(totalPages)+1
scrapedGames = 0
page = 1
last = 0
# This lets us break out of our loop
gamesAvailable = True 
while gamesAvailable:
    #URL To FETCH DATA
    vgURL = 'http://www.vgchartz.com/gamedb/?name=&publisher=&platform=&genre=&minSales=30&results=1000&page=' + str(page) #+ totalGames[1].lstrip()
    req = urllib2.Request(vgURL)
    req.add_unredirected_header('User-Agent','Mozilla/5.0')
    metacritic_url = urllib2.urlopen(req, timeout=50000)
    soup = BeautifulSoup(metacritic_url) 
    rows = soup.find("table", class_="chart").find_all("tr")

    #Fetching and writing Row by Row
    for row in rows:
        data = row.find_all("td")
        if data:
            last = data[0].get_text()
            game["name"] = data[1].get_text()
            game["url"] = data[1].a.get('href')
            game["basename"] = game["url"].rsplit('/', 2)[1]
            game["platform"] = data[2].get_text()
            game["year"] = data[3].get_text()
            game["genre"] = data[4].get_text()
            game["publisher"] = data[5].get_text()
            game["na_sales"] = data[6].get_text()
            game["eu_sales"] = data[7].get_text()
            game["ja_sales"] = data[8].get_text()
            game["rest_sales"] = data[9].get_text()
            game["global_sales"] = data[10].get_text()
            if (float(game["global_sales"]) > 0.00):
                if (game["platform"] in platforms_to_include):
                    metaSoupM = "http://www.metacritic.com/game/" + metacriticPlatform[game["platform"]] +"/"+ game["basename"]
                    fetchDataFromMetaCritic(metaSoupM)
                    writeGame.writerow([game["name"], game["platform"], game["release"], game["genre"], \
								game["publisher"], game["na_sales"], game["eu_sales"], game["ja_sales"], \
								game["rest_sales"], game["global_sales"], game["releaseYear"], \
								game["criticScore"], \
								game["criticCount"], game["userScore"], \
								game["userCount"], game["developer"], \
								game["rating"]])
            scrapedGames += 1
            if(totalGames == int(last)):
                print(" Last Game Extracting ", totalGames, int(last))
                gamesAvailable = False
                
    page += 1 

			

print("Total Games Scraped: ",scrapedGames)
endTime = time.time()
totalTimeForScrapping = endTime - startTime
                                
print("Total time taken for Scrapping: ", totalTimeForScrapping, " secs")
csvFile.close()

            
            
            
            
            
            
            
            
            
            
            
		

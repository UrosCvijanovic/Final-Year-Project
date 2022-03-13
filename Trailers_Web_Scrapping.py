# Import relevant webscraping libraries
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import re
import urllib.request
from urllib.request import Request, urlopen
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome(ChromeDriverManager().install())

df_merged = pd.read_csv('Final_Data_5203.csv', low_memory=False)
# IMPORT THE ORIGINAL DATASET

def ScrapeVideos(movieId):
    output = []
    url = 'https://www.imdb.com/title/' + movieId + '/'
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    mydivs = html_soup.find_all("a", {"class": "ipc-lockup-overlay Slatestyles__SlateOverlay-sc-1t1hgxj-2 fAkXoJ hero-media__slate-overlay ipc-focusable"})
    a_tag_soup = BeautifulSoup(str(mydivs), 'html.parser')
    link_for_download = a_tag_soup.find(href=True)
    print(link_for_download['href'])
    url_of_video = 'https://www.imdb.com/' + link_for_download['href'] + '/'
    print(url_of_video)

    final_response = get(url_of_video)
    final_soup = BeautifulSoup(final_response.text, 'html.parser')
    #print(final_soup)
    test_get = final_soup.find_all('video')
    print(test_get)
    test = final_soup.findAll("div", {"id": "a-page"})
    print(test)
    divs = final_soup.findAll("div", {"class": "jw-media jw-reset"})
    #print(divs)

driver = webdriver.Chrome(ChromeDriverManager().install())
driver = webdriver.Chrome(executable_path='C:\\browserdriver\\chromedriver')
url = 'https://www.imdb.com//video/vi2052129305?playlistId=tt0114709&ref_=tt_ov_vi/'
driver.get(url)

#dwn_link = 'https://imdb-video.media-imdb.com/vi2171864857/1434659607842-pgv4ql-1563573458930.mp4?Expires=1646997679&Signature=tUC3pAJs89MZgpW0Op2UH-PMKzzsORxHPdz4NFmUKpz89SSZez-m~1WNfqe-MayJ4jqGm-1~x76UwFHe5SFqIBRyoTyWIVAWyy1SEnZyW6uwy9T7Ml54VEBRpDvvHidQGmAwFgXmAv6BPmHDjxpdMq3VZoHpapIOBcu5LWPPZJAMTa-SkMjjhMuQpt2lYcHCJ1wxh6vGYd7iZwQcmmaFQBi7YBZRqn4NkcH0hg9UNsprLsjEcD0WwlbpP0EICIeA-NyRJX77CUaxos0h57jXbGNldFp6XHB19w2XkSJl0ZLEjltjhFsy2Iw1NlB4ojokQ0PlZ~d4JyiU7jBSLyvUJw__&Key-Pair-Id=APKAIFLZBVQZ24NQH3KA'


#req = Request(dwn_link, headers={'User-Agent': 'Mozilla/5.0'})
#webpage = urlopen(req, timeout=10).read()

#urllib.request.urlretrieve(dwn_link, 'video_name.mp4')

id = ["tt0114709", "tt0076759"]
for i in id:
    ScrapeVideos(i)


"""
imdb_ids = df_merged['imdb_id']
plotKeywords = []
counter = 0  # counter to keep track of progress
for id in imdb_ids.head(2):
    #print("{} out of {} completed".format(counter, len(imdb_ids)))
    plotKeywords.append(ScrapePlotKeywords(id))
    counter += 1

df_merged['plot_keywords'] = plotKeywords

df_merged = df_merged.drop('index', axis=1)

# Export csv to prevent having to WebScrape again
df_merged.to_csv('merged_dataframe_withPlot_test.csv', index=False)
"""
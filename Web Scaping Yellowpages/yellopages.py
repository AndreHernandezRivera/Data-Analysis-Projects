from bs4 import BeautifulSoup
import requests
import pandas as pd


headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}
# url = 'https://www.yellowpages.com/los-angeles-ca/dentists'

headings=['Name','Address','Info','Category','Links']
df=pd.DataFrame(columns=headings)


for pagenum in range(1,3,1):
    url =F'https://www.yellowpages.ca/search/si/{pagenum}/Barbecue+Equipment+%26+Supplies/Mississauga+ON'
    print(url)
    temp=pd.DataFrame(columns=headings)

    response=requests.get(url,headers=headers)

    soup=BeautifulSoup(response.content,'lxml')

    for i,item in enumerate(soup.select('.listing_right_top_section')):

        
        try:
            # print(item)
            # for el in soup.find_all(attrs={"class": "listing__title--wrap"}):

            for el in item.find_all("a", class_="listing__name--link listing__link jsListingName"):
                print(el['href'])
            link='www.yellowpages.ca'+el['href']
            
            

            temp.loc[i,'Name']=item.find_all("a", class_="listing__name--link listing__link jsListingName")[0].get_text()
            temp.loc[i,'Address']=item.find_all("span", class_="listing__address--full")[0].get_text()
            temp.loc[i,'Info']=item.find_all("article",class_='listing__details__teaser')[0].get_text()
            
            temp.loc[i,'Category']=item.find_all("div", class_="listing__headings")[0].get_text()
            temp.loc[i,'Links']=link
            # temp.loc[i,'Links']=item.find_all("a", class_="listing__name--link listing__link jsListingName")['href'].get_text()
            


            
        except Exception as e:
            #raise e
            print('')
        print('//////')
    df=pd.concat([df, temp])   
    
# df['raw_'] = df['raw_'].replace('\n',' ', regex=True)
df= df.replace('\n','', regex=True)


df.to_csv('yellowpages.csv', encoding='utf-8-sig')    



"""
df['raw_'] = df['raw_'].str.replace('  ',';', regex=False)
"""
        

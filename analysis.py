'''
Pinnacle Bank Monarch Offensive Power Rankings: 2007 thru 2024. Returns list of best to worst seasons.
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import os
import re
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
1. Plate discipline = Walk/SO ratio (BB/SO)
2. Power = Isolated power (2B+2*3B+3*HR)/AB
3. Run contribution = Runs created TB*(H+BB)/(AB+BB)
4. Clutch factor = (RBI + SacF + SacB + SB) / AB

'''



url_main = 'https://www.papillionbaseball.com/teams/default.asp?u=PAPIOPOST32&s=baseball&p=stats&div=158867&viewseas='
seasons = ['Summer_2024','Summer_2023','Summer_2022','Summer_2021','Summer_2019','Summer_2018','Summer_2017','Summer_2016','Summer_2015','Summer_2014','Summer_2013','Summer_2011/2012','Summer_2010','Summer_2009','Summer_2008','Summer_2007']
req_columns = ['No', 'Name', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'HBP', 'BB', 'SO', 'SacB', 'SacF', 'SB', 'CS', 'AVG', 'OBP', 'SLG','Season']


#Given a URL, scrape player statistics for each season
def StatScrape(website):
    #create tempDF to hold current page offensive stats. include everything except final row (last row = totals row)
    tempDF = pd.read_html(website)[2].iloc[:-1]
    tempSeason = re.search(r'(?<=viewseas=).*', website)[0]
    print(f'Collecting {tempSeason.replace("_"," ")} statistics...')

    #Append to master dataframe if columns match up
    if req_columns[:-1] == tempDF.columns.tolist():
        tempDF['Season'] = tempSeason
        return tempDF.values.tolist()
    
    else:
        #figure out which column(s) is not in current page
        for col in req_columns[:-1]:
            if col not in tempDF.columns:
                tempDF[col] = [np.NaN] * len(tempDF)
        
        tempDF['Season'] = tempSeason
        return tempDF[req_columns].values.tolist()

#Takes in raw dataframe and performs some cleaning steps (change NaNs to 0, remove ineligible players)
def CleanStats(stats):
    cStats = stats[req_columns].fillna(0)
    return cStats[cStats['PA']>=20].reset_index()

def AddColumns(stats):
    stats['BB_to_SO_Ratio'] = (stats['BB']/stats['SO'])
    stats['IsoPower'] = (stats['2B']+2*stats['3B']+3*stats['HR'])/stats['AB']
    stats['RC'] = (((stats['H']-stats['2B']+stats['3B']+stats['HR'])+2*stats['2B']+3*stats['3B']+4*stats['HR'])*(stats['H']+stats['BB']))/(stats['AB']+stats['BB'])
    stats['ClutchFactor'] = (stats['RBI'] + stats['SacF'] + stats['SacB'] + stats['SB'])/stats['AB']
    return stats

def Normalize(stats):
    transform = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())
    preprocessor = make_column_transformer((transform, ['BB_to_SO_Ratio','IsoPower','RC','ClutchFactor']))
    return preprocessor.fit_transform(stats[['BB_to_SO_Ratio','IsoPower','RC','ClutchFactor']])


if __name__ == '__main__':
    myList = []
    with mp.Pool(os.cpu_count()) as pool:
        master_data = pool.map(StatScrape, [url_main+s for s in seasons])
        pool.close()
        pool.join()
        for j in master_data:
            for k in j:
                myList.append(k)

    master_data = pd.DataFrame(myList, columns=req_columns)
    cleaned_data = AddColumns(CleanStats(master_data))
    normalized_data = Normalize(cleaned_data)
    player_scores = []

    for i in range(len(normalized_data)):
        player_scores.append([cleaned_data['Name'].iloc[i], cleaned_data['Season'].iloc[i], sum(normalized_data[i])])
    
    pd.DataFrame(player_scores, columns=['Player','Season','JurjScore']).sort_values('JurjScore', ascending=False).to_csv(f"Monarch Baseball JurjScores {datetime.today().strftime('%Y%m%d')}.csv", index=False)


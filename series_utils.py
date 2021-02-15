'''
@info utils to transform time series
@author Rui Henriques
@version 1.0
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import random

def remove_inexistent_days_from_series(original, series, attributes):
    unique_days = original['date'].map(pd.Timestamp.date).unique()
    for attr in attributes:
        series.loc[~series.index.normalize().isin(unique_days), attr] = 'day to drop'
    series = series[series[attr] != 'day to drop']
    return series

def round_date(dt, minutes):
    delta = timedelta(minutes=minutes)
    dat = (datetime(dt.year,dt.month,dt.day,0,0,0) - dt) % delta
    if dat==timedelta(0): return dt
    return dt + dat - delta

def fill_time_series(dataframes,name,idate,fdate,minutes,handledate = False):
    freq = "%dmin" % minutes
    
    if minutes%60 == 0: freq = "%dH" % (minutes/60)
    series = pd.DataFrame(index=pd.date_range(idate, fdate+timedelta(days=1), freq=freq)) #
    series.name = name
    
    for df in dataframes:
        for col in df.columns:
            if col=="date": continue
            newcol = df.name+":"+col
            series[newcol] = 0
            for __, row in df.iterrows():
                if handledate: date = round_date(row["date"],minutes)
                else: date = row["date"]
                series[newcol][date] += row[col]
    return series


def fill_spatial_time_series(dataframes, name, hours_range, minutes, geom_key='path.street_coord', aggregation='mean'):
    series = pd.DataFrame(columns=['date', geom_key])
    series = series.set_index(['date', geom_key])
    series.index = series.index.set_levels([pd.to_datetime(series.index.levels[0]), series.index.levels[1]])

    for df in dataframes:
        if not df.empty:
            df['date'] = df['date'].apply(lambda x: round_date(x, minutes))
            df[geom_key] = df[geom_key].astype(str)
            df = df.reset_index().set_index(['date', geom_key])
            df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])
            if aggregation == 'mean':
                df = df.groupby(level=[0, 1]).mean()
            elif aggregation == 'sum':
                df = df.groupby(level=[0, 1]).sum()
            series = pd.concat([series, df])

    series = series.loc[~(series == 0).all(axis=1)]
    series = series.unstack().between_time('{}:00'.format(hours_range[0]), '{}:00'.format(hours_range[1])).stack()
    series.name = name
    return series

# use the largest block o the time series without any missing value
def select_time_series_to_eval(df):
    nanIndexes = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    
    # if there are no nan on the time series, use all the observations
    if(len(nanIndexes)==0):
        return df
    nanIndexes = np.insert(nanIndexes, 0, 0)
    
#    print("NAN SUM", df.isnull().sum())
#    
#    print("NANINDEXES", nanIndexes, len(nanIndexes))
    
    bestIndexes = []
    bestTimeDif = 0
    
#    print("nanIndexes", nanIndexes[-1], nanIndexes[-2], len(nanIndexes))
#    #print("DIF", df.index[nanIndexes[5+1]] - df.index[nanIndexes[5]])
#    
#    print("XXXXXXXXXXX", df.index[8472])
    
    for i in range(len(nanIndexes) - 1):
#        print("NANINDEXX", nanIndexes[i+1], nanIndexes[i])
        timeDif = df.index[nanIndexes[i+1]] - df.index[nanIndexes[i]]
#        print("TIMEDIF", timeDif)
        parsedToMiliSecondsTimeDif = round(timeDif/ np.timedelta64(1, 's'))
        if(parsedToMiliSecondsTimeDif > bestTimeDif):
            bestTimeDif = parsedToMiliSecondsTimeDif
            bestIndexes = [nanIndexes[i], nanIndexes[i+1]]
            #print("BEST INDEXES", bestIndexes)
        
#    print("LAST STUFF", df.index[bestIndexes[0]+1], df.index[bestIndexes[1]])
    return df.iloc[bestIndexes[0]+1:bestIndexes[1]]
    

# generate artificial missing values on data
def generate_artificial_data(df, settings, dataType, randomSeed):
    
    #print("DATATYPE", settings)
    #print("RANDOMSEED", randomSeed)
    #random.seed(randomSeed)
    
    #result = seasonal_decompose(df.copy(), model='additive')
    pd.set_option('display.max_rows', df.shape[0]+1)
    #print("RESULTADO", result.resid+result.trend+result.seasonal)
    
    artificialPercent = settings[0]
    artificialType = settings[1].lower()
    artificialPeriod = settings[2]
    numSensor = settings[3]
    sameSensor = settings[4]
    
    columnsIndex = []
    if(numSensor == 'all'):
        columnsIndex = np.arange(len(df.columns))
    else:
        aList = list(np.arange(len(df.columns)))
        columnsIndex = random.sample(aList, numSensor) # chose randomly the sensors which will have artificial changed
        
    newData = df.copy()
    
    numberOfNA = int(len(df.index) * artificialPercent)
    if(artificialType == 'mixed'): numberOfNA = round(numberOfNA/2)
    
    if(artificialType == 'sequential' or artificialType == 'mixed'):
        
        divisionPeriods = artificialPeriod + 1
        numberOfNAperPeriod = round(numberOfNA / artificialPeriod)
        numberOfObservationsperPeriod = round(len(df.index) / divisionPeriods)
        
        if(sameSensor): # if we want the sensors to have the same observations with artificial observations
            for i in range(artificialPeriod):
                #print("PERIDO-------------------------", i)
                startRange = numberOfObservationsperPeriod * (i+1)
                endRange = numberOfObservationsperPeriod * (i+2) - numberOfNAperPeriod
                random.seed(randomSeed)
                startingNAIndex = random.randrange(startRange, endRange)
                for sensor in df.columns:
                    if(df.columns.get_loc(sensor) in columnsIndex): # condition for the numSensor to generate artificial observations
                        if(dataType == 'missing'):
                            newData.loc[startingNAIndex:startingNAIndex+numberOfNAperPeriod,sensor] = np.NaN
                        elif(dataType == 'outlier'):
                            #print("STARTINGNAINDEX", startingNAIndex, newData.iloc[startingNAIndex:startingNAIndex+numberOfNAperPeriod].index) #-> TODOD
                            
                            generatedIdx = newData.iloc[startingNAIndex:startingNAIndex+numberOfNAperPeriod].index
                            newData[sensor] = generate_outliers(df, sensor, generatedIdx, randomSeed)
                        else:
                            continue
                    else:
                        continue
                
        else:  # if we DONT want the sensors to have the same observations with NA
            rndSeed = randomSeed
            for sensor in df.columns:
                rndSeed+=10
                if(df.columns.get_loc(sensor) in columnsIndex): # condition for the numSensor to generate artificial observations
                    for i in range(artificialPeriod):
                        startRange = numberOfObservationsperPeriod * (i+1)
                        endRange = numberOfObservationsperPeriod * (i+2) - numberOfNAperPeriod
                        random.seed(rndSeed)
                        startingNAIndex = random.randrange(startRange, endRange)
                        if(dataType == 'missing'):
                            newData.loc[startingNAIndex:startingNAIndex+numberOfNAperPeriod,sensor] = np.NaN
                        elif(dataType == 'outlier'):
                            #print("STARTINGNAINDEX", startingNAIndex, newData.iloc[startingNAIndex:startingNAIndex+numberOfNAperPeriod].index) #-> TODOD
                            #print("BLABLA", newData.loc[startingNAIndex:startingNAIndex+numberOfNAperPeriod,:])
                            generatedIdx = newData.iloc[startingNAIndex:startingNAIndex+numberOfNAperPeriod].index
                            newData[sensor] = generate_outliers(df, sensor, generatedIdx, randomSeed)
                        else:
                            continue
                else:
                    continue
    
    if(artificialType == 'punctual' or artificialType == 'mixed'):
        
        newDataWithoutNA = newData.dropna()
        if(sameSensor): # if we want the sensors to have the same observations with artificial observations
            generatedIdx = np.sort(newDataWithoutNA.sample(n=int(numberOfNA), random_state=randomSeed).index) # generate a list of indexes which will be artificially changed
            for sensor in df.columns:
                if(df.columns.get_loc(sensor) in columnsIndex): # condition for the numSensor to generate artificial observations
                    if(dataType == 'missing'):
                        newData.loc[generatedIdx, sensor] = np.NaN # replace the index values by NaN
                    elif(dataType == 'outlier'):
                        newData.loc[:,sensor] = generate_outliers(df, sensor, generatedIdx, randomSeed)
                    else:
                        continue
                    
                else:
                    continue
        
        else:  # if we DONT want the sensors to have the same observations with artificial observations
            for sensor in df.columns:
                if(df.columns.get_loc(sensor) in columnsIndex): # condition for the numSensor to generate artificial observations
                    #print("RANDOM SEED", randomSeed)
                    
                    randomSeed += (1+df.columns.get_loc(sensor)) * 10
                    #print("RANDOM SEED", randomSeed)
                    generatedIdx = np.sort(newDataWithoutNA.sample(n=numberOfNA, random_state=randomSeed).index) # generate a list of indexes which will be artificially changed
                    
                    #print("GENERATED IDX", generatedIdx)
                    
                    if(dataType == 'missing'):
                        newData.loc[generatedIdx, sensor] = np.NaN # replace the index values by NaN
                    elif(dataType == 'outlier'):
                         newData.loc[:,sensor] = generate_outliers(df, sensor, generatedIdx, randomSeed)
                    else:
                        continue
                    
                else:
                    continue
    
    return newData

# generate a random observation as an outlier
def generate_outliers(df, sensor, generatedIdx, randomSeed):
    
    #print("RANDOM SEED", randomSeed)
    
    for idx in range(len(generatedIdx)):
        #print("IDX", randomSeed*(idx+1)+5)
        #randObservation = df[sensor].sample(n=1, random_state = randomSeed*idx)
        #randObservation = df[sensor].sample(n=1)
        #df.loc[idx, sensor] = random.choice([randObservation.values[0] * 3, randObservation.values[0] / 3])
        
        random.seed(idx)
        
        quantile95 = df[sensor].quantile(q=0.95)
        obsBiggerThanQuantile = df[sensor][df[sensor] > quantile95]
        randObservation = obsBiggerThanQuantile.sample(n=1, random_state = randomSeed*(idx+1)+5)
        #print("RANDOB", randObservation.values[0])
        std = df[sensor].std()
        
        #print("randObservation", randObservation, std)
        
        outlier = randObservation.values[0] + (std * random.uniform(0.3, 1))
        
        #print("OUTLIER", outlier)
        
        df.loc[generatedIdx[idx], sensor] = outlier
        
        #df.loc[idx, sensor] = 1000

    
    #print("NEWDF", newDf)
    #print("DFSENSOR", df.loc[generatedIdx, sensor])
    return df[sensor]

# generate the missing rows to the original series
def generate_missing_rows(df):
    # counts of the difference betweeen observations (gap distribution)
    #res = (pd.Series(df.index[1:]) - pd.Series(df.index[:-1])).value_counts() -> DESCOMENTAR
    x = df.iloc[:10]
    x = x.drop([x.index[6]])
    res = (pd.Series(x.index[1:]) - pd.Series(x.index[:-1])).value_counts()
    print("ORIGINAL", x)
    print("RES", res)
    idx = pd.date_range(x.index[0], x.index[-1], freq=res.index[0])
    print("IDX", idx)
    x = x.reindex(idx, fill_value = np.nan)
    print("NEXT X", x)
    
    return None

if __name__ == '__main__':
    '''today = datetime(2018,10,1,5,45,0)
    index = pd.date_range(today, periods=6, freq='15min')
    columns = ['date','B','C']
    df = pd.DataFrame(columns=columns)
    df['date'] = index 
    df = df.fillna(1) # with 0s rather than NaNs
    df.name = 'ola'
    fill_time_series([df],"mydf",datetime(2018,10,1,5,45,0),datetime(2018,10,1,7,30,0),15)'''

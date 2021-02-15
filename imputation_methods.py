#################
#####Imports#####
#################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import csv
import os
from sklearn import preprocessing

# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# libraries for imputing missing values
import impyute as impy
from missingpy import MissForest
from missingpy import KNNImputer
from statsmodels.tsa.arima_model import ARIMA

import warnings

# Auxiliar Methods

# Delete the input and output csv files used on Python to R
def delete_temp_files():
	os.remove("input_data.csv")
	os.remove("output_data.csv")

# add a column of ones to the data
def add_column(x):
	x = x['value'][:, np.newaxis]
	np.seed = 0
	rand_values = np.random.rand(300,1)
	y = np.concatenate((x, rand_values), axis=1)
	y = pd.DataFrame(y, columns = ['value1', 'value2'])
	return y


# impute missing values according to the selected imputation method on the Processing options panel on GUI
def impute_missing_values(df, sensorDict, imputationMethodMode, randomSeed):
    reconstructedData = None
    reconstructedDataFinal = df.copy()
    imputationMethodMode = imputationMethodMode.lower()
    
    
    for sensor in sensorDict.keys():
        imputeMethod = sensorDict[sensor][0]
        parameters = sensorDict[sensor][1]
        
        #print("SENSOR", sensor)
        #print("IMPUTMETHOD", imputeMethod)
        
        
        if(imputationMethodMode == 'default'):
            reconstructedData = average_method(df.copy(), [], sensor)
        
        else:
            if(imputeMethod == 'Mean'):
                reconstructedData = average_method(df.copy(), [], sensor)
            elif(imputeMethod == 'Median'):
            	reconstructedData = median_method(df.copy(), [], sensor)
            elif(imputeMethod == 'Random sample'):
            	reconstructedData = random_sample_method(df.copy(), [randomSeed], sensor)
            elif(imputeMethod == 'Interpolation'):
            	reconstructedData = interpolation_method(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Locf'):
            	reconstructedData = locf(df.copy(), [], sensor)
            elif(imputeMethod == 'Nocb'):
            	reconstructedData = nocb(df.copy(), [], sensor)
            elif(imputeMethod == 'Moving average'):
            	reconstructedData = moving_average(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Multiple moving average'):
                reconstructedData = multiple_moving_average(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Flexible moving average'):
                reconstructedData = flexible_moving_average(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Random forests'):
                parameters.append(randomSeed)
                reconstructedData = random_forests(df.copy(), parameters, sensor) #so deixar quando a time serie é multi variada
            elif(imputeMethod == 'Expectation maximization'):
            	reconstructedData = mtsdi(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Knn'):
            	reconstructedData = knn(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Mice'):
                parameters.append(randomSeed)
                reconstructedData = mice(df.copy(), parameters, sensor)
            elif(imputeMethod == 'Amelia'):
                reconstructedData = amelia(df.copy(), parameters, sensor)
        
        reconstructedDataFinal[sensor] = reconstructedData[sensor]
        
    return reconstructedDataFinal

'''Imputation Methods'''

#usedSensor is the sensor that comes from the main.py
    

# impute missing values with the mean of the column
def average_method(df, parameters, sensor):
    avg = df[sensor].mean()
    df[sensor] = df[sensor].fillna(avg)
    return df

# impute missing values with the median of the column
def median_method(df, parameters, sensor):
    median = df[sensor].median()
    df[sensor] = df[sensor].fillna(median)
    return df

# impute missing values with a random sample of the column
def random_sample_method(df, parameters, sensor):
    randomSeed = parameters[0]

    randomSampleDS = df[sensor].sample(random_state = randomSeed) #returns a series
    i = randomSeed
    while(np.isnan(randomSampleDS.iloc[0])):
        i+=1
        randomSampleDS = df[sensor].sample(random_state = i)
    randomSample = randomSampleDS.iloc[0]
    #print("RANDOMSAMPLEDS", randomSample)
    df[sensor] = df[sensor].fillna(randomSample)
    
    return df

# impute missing values with linear interpolation method
def interpolation_method(df, parameters, sensor):
    if(parameters[0] == 'polynomial'):
        df[sensor] = df[sensor].astype(float).interpolate(method='polynomial', order=2)
    else:
        df[sensor] = df[sensor].astype(float).interpolate(method='linear')
    
    return df

# impute missing values with locf method
def locf(df, parameters, sensor):
    df[sensor] = df[sensor].fillna(method='ffill')
    return df

# impute missing values with nocb method
def nocb(df, parameters, sensor):
    df[sensor] = df[sensor].fillna(method='backfill')
    return df

# impute missing values with moving average (arithmetic smoothing) method
def moving_average(df, parameters, sensor):
    df[sensor] = df[sensor].rolling(window = parameters[0], min_periods = parameters[1], center = parameters[2]).mean()
    #df[sensor] = df[sensor].rolling(window = parameters[0], min_periods = parameters[1], center = False).mean()
    return df

# impute missing values with moving average (arithmetic smoothing) method
def multiple_moving_average(df, parameters, sensor):
    while(df[sensor].isnull().values.any()):
       df[sensor] = df[sensor].rolling(window = parameters[0], min_periods = parameters[1], center = parameters[2]).mean()
       #df[sensor] = df[sensor].rolling(window = parameters[0], min_periods = parameters[1], center = False).mean()
    return df

# impute missing values with flexible moving average method where the window size expands if there are only missing values
def flexible_moving_average(df, parameters, sensor):
    #print("DF", df)
    windowSize = int((parameters[0]-1)/2)
    impossibleValue = 1000000
    newDf = df[sensor].copy()
    NAindexes = df[sensor].index[df[sensor].isna()]
    '''
    print("WINDOWSIZE", windowSize)
    
    print("NORMAL INDEX", len(NAindexes), NAindexes, NAindexes[0])
    print("BLABAL", df[sensor].index.get_loc(NAindexes[0]))
    print("VALUE", df[sensor].iloc[df[sensor].index.get_loc(NAindexes[0])-1])
    print("INDEX", df[sensor].loc[NAindexes[0]])
    print("FIRST OBSERVATION", df[sensor].iloc[0], df[sensor].iloc[0], df[sensor].iloc[-1])
    print("FIRST OBSERVATION - 1", df[sensor].iloc[0-1])
    '''
    
    for naIdx in range(len(NAindexes)):
        
        #print("NAIDX", NAindexes[naIdx])
        
        #print("NEWDF", newDf)
        
        # switch the NA value for an impossible value and drop all NA values
        tempDf = newDf.copy()
        
        
        tempDf.iloc[df.index.get_loc(NAindexes[naIdx])] = impossibleValue
        newDf.iloc[df.index.get_loc(NAindexes[naIdx])] = impossibleValue
        
        tempDf = newDf.dropna()
      
        print("TEMPDF", tempDf)  
      
        # compute the mean for either right side and left side of the NA, if possible
        numerator = 0
        for windIdx in range(windowSize):
            print("INDEX", tempDf[NAindexes[naIdx]].index)
            #print("LEFT LIMIT", tempDf[sensor].index.get_loc(NAindexes[naIdx]), (windIdx+1))
            leftLimit = tempDf.index.get_loc(NAindexes[naIdx]) - (windIdx+1)
            print("LEFT LIMIT",  leftLimit)
            rightLimit = tempDf.index.get_loc(NAindexes[naIdx])+(windIdx+1)
            print("RIGHR LIMIT", rightLimit) 
            
            #print("LEN DF", len(df.index))
            
            if(leftLimit >= 0):
                #print("LEFT VALUE", tempDf[sensor].iloc[tempDf[sensor].index.get_loc(NAindexes[naIdx])-(windIdx+1)])
                #print("PRE NUM", tempDf[sensor].iloc[tempDf[sensor].index.get_loc(NAindexes[naIdx])-(windIdx+1)])
                numerator += tempDf.iloc[tempDf.index.get_loc(NAindexes[naIdx])-(windIdx+1)]
                
            if(rightLimit <= len(df.index)-1):
                #print("RIGHT VALUE", tempDf[sensor].iloc[tempDf[sensor].index.get_loc(NAindexes[naIdx])+(windIdx+1)])
                numerator += tempDf.iloc[tempDf.index.get_loc(NAindexes[naIdx])+(windIdx+1)]
                
            
        #print("NUMERATOR", numerator)
        total = numerator / (windowSize*2)
        #print("TOTAL", total)
        
        #replace the NA on the final DF
        
        newDf.iloc[df.index.get_loc(NAindexes[naIdx])] = total
        df[sensor] = newDf
    return df

# impute missing values with random forests method
def random_forests(df, parameters, sensor):
    imputer = MissForest(max_iter=parameters[0],n_estimators=parameters[1],
                         min_samples_split=parameters[2],min_samples_leaf=parameters[3], 
                         random_state = parameters[-1], verbose=0)
    
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = imputer.fit_transform(df)
    newDf = pd.DataFrame(x, columns = df.columns, index = df.index)
    return newDf

# impute missing values with expectation maximization (EM) method
def mtsdi(df, parameters, sensor):
    x = impy.em(df.as_matrix().astype(float), loops=parameters[0])
    newDf = pd.DataFrame(x, columns = df.columns, index = df.index)
    return newDf

# impute missing values with nearest neighbor observation method
def knn(df, parameters, sensor):
    imputer = KNNImputer(n_neighbors=parameters[0],weights=parameters[1])
    x = imputer.fit_transform(df.to_numpy())
    newDf = pd.DataFrame(x, columns = df.columns, index = df.index)
    return newDf

# impute missing values with MICE algorithm
def mice(df, parameters, sensor):
    
    if(len(df.columns) < 2): return df
    
    maxValue = []
    minValue = []
    
    if(parameters[3]):
        for dfSensor in df.columns:
            maxValue.append(df[dfSensor].max())
            minValue.append(df[dfSensor].min())
            df[dfSensor] = (df[dfSensor] - df[dfSensor].min()) / (df[dfSensor].max() - df[dfSensor].min())
            
    
	# connect to R
    command = ["Rscript", "imputation_methods.r"]
    df.to_csv('input_data.csv', index=False)
    args = ['mice'] + parameters
    args = [str(i) for i in args]
    subprocess.call(command + args, shell = True)
    outputR = subprocess.check_output(command + args, universal_newlines=True)

	# get info from R
    newData = pd.read_csv("output_data.csv", delimiter=',', encoding = "ISO-8859-1")
    newData = newData.drop(newData.columns[0], axis=1)
    newData.columns = df.columns
    newData.index = df.index
    
    delete_temp_files()
    
    if(parameters[3]):
        for dfSensor in newData.columns:
            colIdx = newData.columns.get_loc(dfSensor)
            newData[dfSensor] = newData[dfSensor]*(maxValue[colIdx]-minValue[colIdx]) + minValue[colIdx]
    
    return newData

# impute missing values with Amelia algorithm
def amelia(df, parameters, sensor):
    #pd.set_option('display.max_rows', 50)
    
    if(len(df.columns) < 2): return df
    
    maxValue = []
    minValue = []
    
    if(parameters[3]):
        for dfSensor in df.columns:
            maxValue.append(df[dfSensor].max())
            minValue.append(df[dfSensor].min())
            df[dfSensor] = (df[dfSensor] - df[dfSensor].min()) / (df[dfSensor].max() - df[dfSensor].min())

	# connect to R
    command = ["Rscript", "imputation_methods.r"]
    df.to_csv('input_data.csv', index=False)
    args = ['amelia'] + parameters[:-1]
    args = [str(i) for i in args]
    subprocess.call(command + args, shell = True)
    outputR = subprocess.check_output(command + args, universal_newlines=True)

	# get info from R
    newData = pd.read_csv("output_data.csv", delimiter=',', encoding = "ISO-8859-1")
    newData = newData.drop(newData.columns[0], axis=1)
    newData.columns = df.columns
    newData.index = df.index
    delete_temp_files()
    
    if(parameters[3]):
        for dfSensor in newData.columns:
            colIdx = newData.columns.get_loc(dfSensor)
            newData[dfSensor] = newData[dfSensor]*(maxValue[colIdx]-minValue[colIdx]) + minValue[colIdx]

    return newData

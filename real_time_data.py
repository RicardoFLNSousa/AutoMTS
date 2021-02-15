# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:14:19 2020

@author: Ricardo
"""

import random
import numpy as np
from threading import Thread, Event
from time import sleep
import imputation_methods, series_utils, outlier_detection_methods, evaluation
import pandas as pd

import warnings


###############
## CONSTANTS ##
##############

imputation_methods_parameters = {'Mean': [],
                                 'Median': [],
                                 'Random sample': [],
                                 'Interpolation': ['linear'],
                                 'Locf': [],
                                 'Nocb': [],
                                 'Moving average': [5,1,False],
                                 'Multiple moving average': [5,1,False],
                                 'Random forests': [10,100,2,1],
                                 'Expectation maximization': [50],
                                 'Knn': [5,'uniform'],
                                 'Mice': [5,'pmm',5,False],
                                 'Amelia': [5,0.05,100,False]}

outlier_methods_parameters = {'Standard deviation': [],
                              'Inter quartile range': [],
                              'Isolation forests': [100,1],
                              'Local outlier factor': [20, 'minkowski'],
                              'Dbscan': [0.5,5,'euclidean'],
                              'K-means': [],
                              'Sax': ['uniform']}

###################
## AUX FUNCTIONS ##
###################


# add a new DateTime column to the dataframe
def add_time_on_columns(df):
    # split the original date time columns by date and time
    df_dt = pd.DataFrame(df.iloc[:,0].str.split(" ").tolist(), columns = ['date', 'time'])
    # the times that are null fill with 00:00:00
    df_dt['time'] = df_dt['time'].fillna('00:00:00')
    # join the date and time columns into one single column
    df_dt = df_dt.apply(lambda row: row.date + " " + row.time, axis = 1)
    # add the new column to the dataframe
    df.insert(loc = 2, column = 'DateTime', value = df_dt.to_list())
    return df

def get_series(nRows):
    df = pd.read_excel('./Datasets/barreiro_telegestao.xls')
            
    # get the original columns names from the excell file
    df.columns = df.iloc[1, :].tolist() 
    
    #remove the initial trash rows
    df = df.iloc[3:] 
    df = add_time_on_columns(df)
    # use the DateTime column has index, with the format YYYY-mm-dd HH:MM:SS
    df.index = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S') 
    
    #remove the initial trash columns including the original and new datetimes
    #df = df.iloc[:672, 10]
    df = df.iloc[:nRows, 10:12]#df.iloc[:10, 10:12]
    
    if(isinstance(df, pd.Series)):
        frame = {'Sensor': df}
        df = pd.DataFrame(frame)
    
    return df

# 
def detect_outliers(prevObservations, nextObservations, sensorsDict, finalSeries, sharedSeries):
    # get the last observations from the finalSeries (which is the series that is already pre-processed)
    prevValuesDf = finalSeries.tail(prevObservations)
    # get the first observations of the sharedSeries
    nextValuesDf = sharedSeries.head(nextObservations)
    currObservationIdx = nextValuesDf.index[0]
    concatedDf = pd.concat([prevValuesDf, nextValuesDf])
    pd.set_option('display.max_rows', concatedDf.shape[0]+1)
    #print("concatedDf", concatedDf)
    reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.detect_outliers(concatedDf, sensorsDict, 'none', 0)
    #print("reconstructedDataFinal.tail-----------------------", reconstructedDataFinal, reconstructedDataFinalNan)
    #print("--------------------------------------")
    pd.set_option('display.max_rows', concatedDf.shape[0]+1)

        
    
    #firstNAIdx = reconstructedDataFinalNan.isnull().any(1).nonzero()[0][0]

    return reconstructedDataFinal.loc[currObservationIdx], reconstructedDataFinalNan.loc[currObservationIdx]


# prevObservations is the previous observations before the NA and nextObservations is the next observations after the NA
def impute_missing_values(prevObservations, nextObservations, imputationSensorsDict, finalSeries, sharedSeries):
    # get the last observations from the finalSeries (which is the series that is already pre-processed)
    prevValuesDf = finalSeries.tail(prevObservations)
    # get the first observations of the sharedSeries
    nextValuesDf = sharedSeries.head(nextObservations)
    currObservationIdx = nextValuesDf.index[0]
    #print("NEEXTVALUESDF", nextValuesDf)
    # create a concatedDf with both previous and next observations
    concatedDf = pd.concat([prevValuesDf, nextValuesDf])
    #print("HAS INDEX", concatedDf.isnull().any(1).nonzero()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        firstNAIdx = concatedDf.isnull().any(1).nonzero()[0][0]
        
    reconstructedData = imputation_methods.impute_missing_values(concatedDf, imputationSensorsDict, 'none', 0)
    return reconstructedData.loc[currObservationIdx]

# usedObservations are the n previous observations before the NA
def impute_missing_values_previous(usedObservations):
    prevValuesDf = finalSeries.iloc[-usedObservations+1:]
    reconstructedData = imputation_methods.impute_missing_values(prevValuesDf, imputationSensorsDict, 'none', 0)
    return reconstructedData.iloc[-1]

# evaluation of the real time data. dataType is either missings or outliers
# origData is the original data. predData is the data after the pre-processing. artificialData is the data after the missings 
# or outliers were generated
def evaluate(dataType, origData, predData, artificialData):
    for sensor in origData.columns:
        if(dataType == 'missing'):
            indexNA = artificialData.loc[artificialData[sensor].isna(), :].index
            mse, rmse, mae, smape = evaluation.evaluate_singular(origData, predData, artificialData, indexNA, sensor)
            print("EVALUATION OF SENSOR", sensor, ":", mse,rmse,mae,smape)

#############
## THREADS ##
#############

# read the values from the shared series and write them on a final series with the NA removed. sleep of 0.5 seconds
def read_values(sharedList):
    currIdx = 0
    while True:
        
        print("LISTA PARTILHADA", sharedList)
        
        # using both previous and the buffer observations which are also the next observations
        
        # if the buffer has already maxBufferSize elements inside or we are in the last maxBufferSize observations
        if(len(sharedList) > maxBufferSize or (originalLen - len(finalSeries)) <= maxBufferSize):
            
            
            rowValues = sharedList.iloc[0]
            
            # print("ROWVALUES", rowValues)
            # # check if any of the sensors has outliers (NAN when outlier)
            # reconstructedDataFinal, reconstructedDataFinalNan = detect_outliers(2, 2, sensorsDict, finalSeries, sharedSeries)
            # print("RECONDATANAN", reconstructedDataFinalNan)
            # #rowValues = reconstructedDataFinalNan
            
            # check if any of the sensors has missings
            if(rowValues.isnull().values.any()):
                print("rowValues", rowValues, "-----", np.isnan(rowValues['Pressão Méd.']))
                rowValues = impute_missing_values(10,6, imputationSensorsDict, finalSeries, sharedSeries)
                
            
            finalSeries.loc[sharedSeries.index[0]] = rowValues
            sharedList.drop(sharedList.index[0], inplace=True)
          
        # using only the previous observations. only for methods dependable of the previous observations
        '''
        print("LISTA PARTILHADA", sharedList)
        if(prevLen != len(sharedList.index)):
           
            rowValues = sharedList.iloc[currIdx]
            
            if(rowValues.isnull().values.any()):
                print("É NAN")
                rowValues = impute_missing_values_previous(10)
                 
            
            imputedSeries.loc[sharedSeries.index[prevLen]] = rowValues
            #prevLen = len(sharedSeries.index)
            currIdx += 1
            print("PREVLEN", prevLen)
        ''' 
        if(originalLen == len(finalSeries)):
            break
        sleep(.5)
        
# write the values of the serie from the original series to the list shared by the two threads . sleep of 1 second
def write_values(series):
    while True:
        if(len(series.index)==0): break
        sharedSeries.loc[series.index[0]] = series.iloc[0]
        #sharedBuffer.loc[series.index[0]] = series.iloc[0]
        series.drop(series.index[0], inplace=True)
        sleep(1)
        
##########
## MAIN ##
##########

if __name__ == "__main__":
    
    # settings of the missing values: percentNA, typeNA, artificialPeriod, numSensor, sameSensor
    settingsNA = [0.1,'punctual',2,'all',False]
    
    # outlier detection method and respective parameters
    outlierMethod = 'Isolation forests' #'Inter quartile range' #'Standard deviation'
    outlierParameters = outlier_methods_parameters[outlierMethod]
    
    # imputation method and respective parameters (if it has) WARNING: if a multivariate method is used and the serie has 
    # only 1 sensor, it will crash the code. check the get_series() function in this script to check how many sensors are being used
    imputationMethod = 'Locf' #'Random forests'
    imputationParameters = imputation_methods_parameters[imputationMethod]
    
    # event to stop the final thread
    event = Event()
    
    # initial number of observations that we want to have at the start
    initObservations = 10
    
    # max number of observations used
    maxObservations = 20
    
    # original series
    originalSeries = get_series(maxObservations)
    
    # the first initObservations observations of the series (which are the ones that we dont want to generate missings or outliers)
    initOriginalSeries = originalSeries.iloc[:initObservations]
    
    # the rest of the observations of the series (which are the ones that we want to generate missings or outliers)
    restOriginalSeries = originalSeries.iloc[initObservations:]
    
    # length of the original series used has stop condition in the readValues() method
    originalLen = len(originalSeries)

    # original series with artificial genetared missing values
    restOriginalSeriesWithArtificial = series_utils.generate_artificial_data(restOriginalSeries, settingsNA, 'missing', 0)
    
    # join the initOriginalSeries and the restOriginalSeries with generated missings or outliers
    originalSeries = pd.concat([initOriginalSeries, restOriginalSeriesWithArtificial])
    
    pd.set_option('display.max_rows', originalSeries.shape[0]+1)
    
    print("PRE ORIGINAL SERIES", restOriginalSeries)
    
    # create an empty series shared between the two threads, with the same columns(sensors) as the original series
    sharedSeries = pd.DataFrame(columns=originalSeries.columns)
    #sharedSeries = originalSeries.iloc[:initObservations]
    
    # create an empty series for the detected outliers and imputed missing values, with the same columns(sensors) as the original series
    finalSeries = pd.DataFrame(columns=originalSeries.columns)
    #imputedSeries = sharedSeries
    
    # buffer with the n observations
    maxBufferSize = 5
    
    # dict used in the detect_outliers() funciton from the outlier_detection_methods.py script
    outliersSensorsDict = dict.fromkeys(originalSeries.columns, [outlierMethod, outlierParameters])
    
    # dict used in the impute_missing_values() function from the imputation_methods.py script
    imputationSensorsDict = dict.fromkeys(originalSeries.columns, [imputationMethod, imputationParameters])
    
    print(originalSeries) 
    
    t1 = Thread(target=write_values, args=(originalSeries, ))
    t2 = Thread(target=read_values, args=(sharedSeries, ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    pd.set_option('display.max_rows', finalSeries.shape[0]+1)
    print("FINAL SERIES", finalSeries)
    evaluate('missing', restOriginalSeries, finalSeries.iloc[initObservations:], restOriginalSeriesWithArtificial)
    
    
    
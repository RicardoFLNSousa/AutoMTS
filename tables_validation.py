import pandas as pd
import imputation_methods, series_utils, outlier_detection_methods
import warnings
import evaluation, evaluation_outliers
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from matplotlib.ticker import MaxNLocator


datasetType = ['telegestao']#, 'telemetria']
#company = ['barreiro', 'beja', 'infraquinta']
company = ['beja']
sensors = ['Pressao', 'Caudal']
#artificialGenerations = ['punctual,0.02', 'punctual,0.1', 'sequential,0.02', 'sequential,0.1']
#artificialGenerations = ['punctual,0.02', 'punctual,0.1']
artificialGenerations = ['sequential,0.02', 'sequential,0.1']

imputationMethods = ['Mean', 'Median', 'Random sample', 'Interpolation', 'Locf', 'Nocb', 'Moving average', 
                     'Random forests', 'Expectation maximization', 'Knn', 'Mice','Amelia']

outlierDetectionMethods = ['Standard deviation', 'Inter quartile range', 'Isolation forests', 'Local outlier factor',
                           'Dbscan', 'Sax']


sensorDictMeanBarr = {'Pressao': ['Mean', None], 'Caudal': ['Mean', None]}
sensorDictMedianBarr = {'Pressao': ['Median', None], 'Caudal': ['Median', None]}
sensorDictRandomSampleBarr = {'Pressao': ['Random sample', None], 'Caudal': ['Random sample', None]}
sensorDictInterpolationBarr = {'Pressao': ['Interpolation', ['linear']], 'Caudal': ['Interpolation', ['linear']]}
sensorDictLocfBarr = {'Pressao': ['Locf', None], 'Caudal': ['Locf', None]}
sensorDictNocbBarr = {'Pressao': ['Nocb', None], 'Caudal': ['Nocb', None]}
sensorDictMovingAverageBarr = {'Pressao': ['Moving average', [5,1,False]], 'Caudal': ['Moving average', [5,1,False]]}
sensorDictFlexibleMovingAverageBarr = {'Pressao': ['Flexible moving average', [3]], 'Caudal': ['Flexible moving average', [3]]}

sensorDictRandomForestsBarr = {'Pressao': ['Random forests', [10,100,2,1]], 'Caudal': ['Random forests', [10,100,2,1]]}
sensorDictExpectationMaxBarr = {'Pressao': ['Expectation maximization', [50]], 'Caudal': ['Expectation maximization', [50]]}
sensorDictKnnBarr = {'Pressao': ['Knn', [5,'uniform']], 'Caudal': ['Knn', [5,'uniform']]}
sensorDictMiceBarr = {'Pressao': ['Mice', [5,'pmm',5,False]], 'Caudal':['Mice', [5,'pmm',5,False]]}
sensorDictAmeliaBarr = {'Pressao': ['Amelia', [5,0.05,100,False]], 'Caudal': ['Amelia', [5,0.05,100,False]]}

dictImputationMethodsBarreiro = {'Mean': sensorDictMeanBarr,
                                 'Median': sensorDictMedianBarr,
                                 'Random sample': sensorDictRandomSampleBarr,
                                 'Interpolation': sensorDictInterpolationBarr,
                                 'Locf': sensorDictLocfBarr,
                                 'Nocb': sensorDictNocbBarr,
                                 'Moving average': sensorDictMovingAverageBarr,
                                 'Flexible moving average': sensorDictFlexibleMovingAverageBarr,
                                 'Random forests': sensorDictRandomForestsBarr,
                                 'Expectation maximization': sensorDictExpectationMaxBarr,
                                 'Knn': sensorDictKnnBarr,
                                 'Mice': sensorDictMiceBarr,
                                 'Amelia': sensorDictAmeliaBarr}

sensorDictMeanBeja = {'Pressao': ['Mean', None], 'Caudal': ['Mean', None]}
sensorDictMedianBeja = {'Pressao': ['Median', None], 'Caudal': ['Median', None]}
sensorDictRandomSampleBeja = {'Pressao': ['Random sample', None], 'Caudal': ['Random sample', None]}
sensorDictInterpolationBeja = {'Pressao': ['Interpolation', ['linear']], 'Caudal': ['Interpolation', ['linear']]}
sensorDictLocfBeja = {'Pressao': ['Locf', None], 'Caudal': ['Locf', None]}
sensorDictNocbBeja = {'Pressao': ['Nocb', None], 'Caudal': ['Nocb', None]}
sensorDictMovingAverageBeja = {'Pressao': ['Moving average', [5,1,False]], 'Caudal': ['Moving average', [5,1,False]]}
sensorDictFlexibleMovingAverageBeja = {'Pressao': ['Flexible moving average', [3]], 'Caudal': ['Flexible moving average', [3]]}

sensorDictRandomForestsBeja = {'Pressao': ['Random forests', [10,100,2,1]], 'Caudal': ['Random forests', [10,100,2,1]]}
sensorDictExpectationMaxBeja = {'Pressao': ['Expectation maximization', [50]], 'Caudal': ['Expectation maximization', [50]]}
sensorDictKnnBeja= {'Pressao': ['Knn', [5,'uniform']], 'Caudal': ['Knn', [5,'uniform']]}
sensorDictMiceBeja = {'Pressao': ['Mice', [5,'pmm',5,False]], 'Caudal':['Mice', [5,'pmm',5,False]]}
sensorDictAmeliaBeja = {'Pressao': ['Amelia', [5,0.05,100,False]], 'Caudal': ['Amelia', [5,0.05,100,False]]}

dictImputationMethodsBeja = {'Mean': sensorDictMeanBeja,
                            'Median': sensorDictMedianBeja,
                            'Random sample': sensorDictRandomSampleBeja,
                            'Interpolation': sensorDictInterpolationBeja,
                            'Locf': sensorDictLocfBeja,
                            'Nocb': sensorDictNocbBeja,
                            'Moving average': sensorDictMovingAverageBeja,
                            'Flexible moving average': sensorDictFlexibleMovingAverageBeja,
                            'Random forests': sensorDictRandomForestsBeja,
                            'Expectation maximization': sensorDictExpectationMaxBeja,
                            'Knn': sensorDictKnnBeja,
                            'Mice': sensorDictMiceBeja,
                            'Amelia': sensorDictAmeliaBeja}


sensorDictStdDevBarr = {'Pressao': ['Standard deviation', None], 'Caudal': ['Standard deviation', None]}
sensorDictIntQuaRanBarr = {'Pressao': ['Inter quartile range', None], 'Caudal': ['Inter quartile range', None]}
sensorDictIsoForBarr = {'Pressao': ['Isolation forests', [100,1]], 'Caudal': ['Isolation forests', [100,1]]}
sensorDictLOFBarr = {'Pressao': ['Local outlier factor', [20,'minkowski']], 'Caudal': ['Local outlier factor', [20,'minkowski']]}
sensorDictDBScanBarr = {'Pressao': ['Dbscan', [0.5,5,'euclidean']], 'Caudal': ['Dbscan', [0.5,5,'euclidean']]}
sensorDictKmeansBarr = {'Pressao': ['K-means', []], 'Caudal': ['K-means', []]}
sensorDictSaxBarr = {'Pressao': ['Sax', ['uniform']], 'Caudal': ['Sax', ['uniform']]}

dictOutlierMethodsBarr = {'Standard deviation': sensorDictStdDevBarr,
                        'Inter quartile range': sensorDictIntQuaRanBarr,
                        'Isolation forests': sensorDictIsoForBarr,
                        'Local outlier factor': sensorDictLOFBarr,
                        'Dbscan': sensorDictDBScanBarr,
                        'K-means': sensorDictKmeansBarr,
                        'Sax': sensorDictSaxBarr}


generationSettings = [2, 'all', False]

# number of instances used in each of the series 
numOfObservationsUsed = 200 #672 equal 4 weeks of observations with 1 hour granulatiry

# number of evaluations on multiple_evaluation function
evalIter = 30

# add a new DateTime column to the dataframe of barreiro
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

def integration_func(data):
    return  integrate.simps(data) 

# upload all files used for the tables
def upload_files(numOfInstances):
    dfBarreiro = pd.read_excel('Datasets/barreiro_telegestao.xls')
    # get the original columns names from the excell file
    dfBarreiro.columns = dfBarreiro.iloc[1, :].tolist() 
    # remove the initial trash rows
    dfBarreiro = dfBarreiro.iloc[3:] 
    dfBarreiro = add_time_on_columns(dfBarreiro)
    # use the DateTime column has index, with the format YYYY-mm-dd HH:MM:SS
    dfBarreiro.index = pd.to_datetime(dfBarreiro['DateTime'], format='%d/%m/%Y %H:%M:%S') 
    # remove the initial trash columns including the original and new datetimes
    dfBarreiro = dfBarreiro.iloc[:, 3:]
    # only use the Pressao Méd and Caudal Méd columns
    dfBarreiro = dfBarreiro.loc[:, dfBarreiro.columns.intersection(['Pressão Méd.','Caudal Méd.'])]
    # rename the colummns to Pressao and Caudal
    dfBarreiro = dfBarreiro.rename(columns={'Pressão Méd.': 'Pressao', 'Caudal Méd.': 'Caudal'})
    # use only the first numOfInstaces of the serie
    dfBarreiro = dfBarreiro.iloc[:672]
    
    ############################################################################################
    
    dfBeja = pd.read_excel('Datasets/beja_telegestao.xlsx')
    # invert the order of the rows
    dfBeja = dfBeja.iloc[::-1]
    # use only a partial amount of the data (1 month)
    dfBeja = dfBeja.iloc[50:21800]
    # set the DateTime column as index
    dfBeja.index = dfBeja['DateAndTime']
    # filter the rows of the caudal and pressao sensors
    caudal = dfBeja.loc[dfBeja['Descritivo'] == 'Beja - Câmara Zona Alta ZMC1 Caudal']['Value']
    pressao = dfBeja.loc[dfBeja['Descritivo'] == 'Beja - Câmara Zona Alta ZMC1 Pressao']['Value']
    d = {'Pressao': pressao.values, 'Caudal': caudal.values}
    dfBeja = pd.DataFrame(d, index = pressao.index)
    # fill the series if have NA using the bfill method #TODO melhorar esta parte para utilizar o maior pedaço da serie sem missings
    #dfBeja = dfBeja.resample('60min').mean().bfill()
    
    finalDfBeja = dfBeja.copy()
    finalDfBeja = finalDfBeja.resample('15min').mean()
    
    caudal = dfBeja['Caudal'].resample('15min').apply(integration_func).bfill()
    pressao = dfBeja['Pressao'].resample('15min').mean().bfill()
    
    finalDfBeja['Caudal'] = caudal
    finalDfBeja['Pressao'] = pressao
    
    # use only the first numOfInstaces of the serie
    #dfBeja = dfBeja.iloc[:numOfInstances]
    
    
    # ############################################################################################
    # dfInfraquinta = pd.read_excel('Datasets/infraquinta_telegestao.xlsx')
    # # set the index
    # dfInfraquinta.index = dfInfraquinta['Row Labels']
    # # drop the columns which corresponded as the index
    # dfInfraquinta = dfInfraquinta.drop(columns=['Row Labels'])
    # # rename the colummns to Pressao and Caudal
    # dfInfraquinta = dfInfraquinta.rename(columns={'QV Sonda de Pressao': 'Pressao', 'QV Caudal': 'Caudal'})
    # # fill the series if have NA using the bfill method #TODO melhorar esta parte para utilizar o maior pedaço da serie sem missings
    # dfInfraquinta = dfInfraquinta.resample('60min').mean().bfill()
    # # use only the first numOfInstaces of the serie
    # dfInfraquinta = dfInfraquinta.iloc[:numOfInstances]
    
    return dfBarreiro, finalDfBeja

def evaluate_multiple_NA(origData, imputationEvaluationSettings, sensorsDict, imputationMethodMode, evalIter): 
    
    # this allMetric is the metric for each iteration
    allMse = []
    allRmse = []
    allMae = []
    allSmape = []
    allPercentOfImputedNA = []
    
    timeSeriesToEval = series_utils.select_time_series_to_eval(origData)
    
    for i in range(evalIter):
        print("EVAL ITER", i)
        dfWithMissing = series_utils.generate_artificial_data(timeSeriesToEval.copy(), imputationEvaluationSettings, 'missing', i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfMissingImputed = imputation_methods.impute_missing_values(dfWithMissing.copy(), sensorsDict, imputationMethodMode, i)
        #print("DFWITHMISSING", dfWithMissing)
        sensorMse = []
        sensorRmse = []
        sensorMae = []
        sensorSmape = []
        sensorPercentOfImputedNA = []
        
        for sensor in origData.columns:   
            #print("SENSOR", sensor)
            index = dfWithMissing.loc[dfWithMissing[sensor].isna(), sensor].index #indexes that will be used on the evaluation (which are the ones which had NA)
            #print("OUTMISSING", dfWithMissing)
            #print("INDEX", index)
            
            percentOfImputedNA = 1 - (dfMissingImputed[sensor].isna().sum()/dfWithMissing[sensor].isna().sum())
            
            #print("PERCENT OF IMPUTEDNA", percentOfImputedNA)
            
            if(len(index) == 0): #if there is no NA in the sensor, continue
                continue
            
            mse, rmse, mae, smape = evaluation.evaluate_singular(timeSeriesToEval, dfMissingImputed, dfWithMissing, index, sensor)
            
            #print("EVAL", mse, rmse, mae, smape)
            
            sensorMse.append(mse)
            sensorRmse.append(rmse)
            sensorMae.append(mae)
            sensorSmape.append(smape)
            sensorPercentOfImputedNA.append(percentOfImputedNA)
        
        allMse.append(sensorMse)
        allRmse.append(sensorRmse)
        allMae.append(sensorMae)
        allSmape.append(sensorSmape)
        allPercentOfImputedNA.append(sensorPercentOfImputedNA)

    return np.array(allRmse), np.array(allMae), np.array(allSmape), np.array(allPercentOfImputedNA)

def evaluate_multiple_outliers(origData, outliersEvaluationSettings, sensorsDict, imputationMethodMode, evalIter): 
    
    # this allMetric is the metric for each iteration
    allAccuracy = []
    allPrecision = []
    allRecall = []
    allF1Score = []
    #allPercentOfImputedNA = []
    
    reconstructedDataFinal = origData.copy()
    reconstructedDataFinalNan = origData.copy()
    
    timeSeriesToEval = series_utils.select_time_series_to_eval(origData)
    
    for i in range(evalIter):
        print("EVAL ITER", i)
        dfWithOutliers = series_utils.generate_artificial_data(timeSeriesToEval.copy(), outliersEvaluationSettings, 'outlier', i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), sensorsDict, imputationMethodMode, i)
        sensorAccuracy = []
        sensorPrecision = []
        sensorRecall = []
        sensorF1Score = []
        #sensorPercentOfImputedNA = []

        for sensor in timeSeriesToEval.columns:     
            detectedOutliersIndex = reconstructedDataFinalNan.loc[reconstructedDataFinalNan[sensor].isna(), :].index
            
            dfGeneratedWithOutliersIndex = timeSeriesToEval[dfWithOutliers[sensor] != timeSeriesToEval[sensor]].index #indexes that will be used on the evaluation (which are the ones which originally had outliers)
            
            #percentOfImputedNA = 1 - (dfMissingImputed[sensor].isna().sum()/dfWithOutliers[sensor].isna().sum())
            
            if(len(dfGeneratedWithOutliersIndex) == 0): #if there is no NA in the sensor, continue
                continue
            
            accuracy, precision, recall, f1Score = evaluation_outliers.evaluate_singular(dfWithOutliers[sensor].copy(), dfGeneratedWithOutliersIndex, detectedOutliersIndex)
            sensorAccuracy.append(accuracy)
            sensorPrecision.append(precision)
            sensorRecall.append(recall)
            sensorF1Score.append(f1Score)
            #sensorPercentOfImputedNA.append(percentOfImputedNA)
        
        allAccuracy.append(sensorAccuracy)
        allPrecision.append(sensorPrecision)
        allRecall.append(sensorRecall)
        allF1Score.append(sensorF1Score)
        #allPercentOfImputedNA.append(sensorPercentOfImputedNA)
    
    return np.array(allAccuracy), np.array(allPrecision), np.array(allRecall), np.array(allF1Score)
    
    
def generate_tables_NA(dfBarreiro, dfBeja):
    
    tablesList = []
    
    for comp in company:
        
        usingDf = dfBarreiro if(comp=='barreiro') else dfBeja #if(comp=='beja') else dfInfraquinta
        
        dfStats = pd.DataFrame(columns=['Name', 'RMSE', 'MAE', 'SMAPE', 'ImputedNA'])
        print("DFSTATS INICIO", dfStats)
        
        for artificialGeneration in artificialGenerations:
            
            genSettings = generationSettings.copy()
            
            typeNA = artificialGeneration.split(',')[0]
            percentNA = artificialGeneration.split(',')[1]
            
            genSettings.insert(0, float(percentNA))
            genSettings.insert(1, typeNA)
            
            print("GENERATION SETTINGS", genSettings)
            
            dfTempPressao = pd.DataFrame(index=[imputationMethods], columns=['RMSE', 'MAE', 'SMAPE', 'ImputedNA'])
            dfTempCaudal = pd.DataFrame(index=[imputationMethods], columns=['RMSE', 'MAE', 'SMAPE', 'ImputedNA'])
            
            for impMethod in imputationMethods:
                print("IMP METHOD", impMethod)
                
                sensorDict = dictImputationMethodsBarreiro[impMethod] if comp=='barreiro' else dictImputationMethodsBeja[impMethod]
                
                allRmse, allMae, allSmape, allPercentOfImputedNA = evaluate_multiple_NA(usingDf, genSettings, sensorDict, '', evalIter)
                
                for s in range(len(sensors)):
                    sensorName = sensors[s]
                    
                    if(sensorName == 'Pressao'):
                        sensorRmse = allRmse[:,s]
                        sensorMae = allMae[:,s]
                        sensorSmape = allSmape[:,s]
                        sensorImputedNA = allPercentOfImputedNA[:,s]
                        
                        dfTempPressao.loc[impMethod,'RMSE'] = str(round(np.mean(sensorRmse),3)) + ' $\pm$ ' + str(round(np.std(sensorRmse),3))
                        dfTempPressao.loc[impMethod,'MAE'] = str(round(np.mean(sensorMae),3)) + ' $\pm$ ' + str(round(np.std(sensorMae),3))
                        dfTempPressao.loc[impMethod,'SMAPE'] = str(round(np.mean(sensorSmape),3)) + ' $\pm$ ' + str(round(np.std(sensorSmape),3))
                        dfTempPressao.loc[impMethod, 'ImputedNA'] = str(round(np.mean(sensorImputedNA),3)) + ' $\pm$ ' + str(round(np.std(sensorImputedNA),3))
                        
                        statName = comp + '_' + impMethod + '_' + 'pressao' + '_' + typeNA + '_' + percentNA
                        dfStats = dfStats.append({'Name': statName, 'RMSE': sensorRmse, 'MAE': sensorMae, 
                                                  'SMAPE': sensorSmape, 'ImputedNA': sensorImputedNA}, ignore_index=True)

                        
                    elif(sensorName == 'Caudal'):
                        sensorRmse = allRmse[:,s]
                        sensorMae = allMae[:,s]
                        sensorSmape = allSmape[:,s]
                        sensorImputedNA = allPercentOfImputedNA[:,s]
                        
                        dfTempCaudal.loc[impMethod,'RMSE'] = str(round(np.mean(sensorRmse),3)) + ' $\pm$ ' + str(round(np.std(sensorRmse),3))
                        dfTempCaudal.loc[impMethod,'MAE'] = str(round(np.mean(sensorMae),3)) + ' $\pm$ ' + str(round(np.std(sensorMae),3))
                        dfTempCaudal.loc[impMethod,'SMAPE'] = str(round(np.mean(sensorSmape),3)) + ' $\pm$ ' + str(round(np.std(sensorSmape),3))
                        dfTempCaudal.loc[impMethod, 'ImputedNA'] = str(round(np.mean(sensorImputedNA),3)) + ' $\pm$ ' + str(round(np.std(sensorImputedNA),3))
                        
                        statName = comp + '_' + impMethod + '_' + 'caudal' + '_' + typeNA + '_' + percentNA
                        dfStats = dfStats.append({'Name': statName, 'RMSE': sensorRmse, 'MAE': sensorMae, 
                                                  'SMAPE': sensorSmape, 'ImputedNA': sensorImputedNA}, ignore_index=True)

                        
            captionNamePressao = comp + ' ' + 'pressao' + ' ' + typeNA + ' ' + ('2\%NA' if percentNA == '0.02' else '10\%NA')
            latexPressao = dfTempPressao.to_latex()
            tablesList.append([captionNamePressao, latexPressao])
            
            captionNameCaudal = comp + ' ' + 'caudal' + ' ' + typeNA + ' ' + ('2\%NA' if percentNA == '0.02' else '10\%NA')
            latexCaudal = dfTempCaudal.to_latex()
            tablesList.append([captionNameCaudal, latexCaudal])
    
    dfStats.to_csv('Validation/stats.csv')
    
    return tablesList

def generate_tables_out(dfBarreiro, dfBeja):
    
    tablesList = []
    
    for comp in company:
        
        usingDf = dfBarreiro if(comp=='barreiro') else dfBeja #if(comp=='beja') else dfInfraquinta
        
        dfStats = pd.DataFrame(columns=['Name', 'F1Score', 'Accuracy', 'Precision', 'Recall'])
        print("DFSTATS INICIO", dfStats)
        
        for artificialGeneration in artificialGenerations:
            
            genSettings = generationSettings.copy()
            
            typeOut = artificialGeneration.split(',')[0]
            percentOut = artificialGeneration.split(',')[1]
            
            genSettings.insert(0, float(percentOut))
            genSettings.insert(1, typeOut)
            
            print("GENERATION SETTINGS", genSettings)
            
            dfTempPressao = pd.DataFrame(index=[outlierDetectionMethods], columns=['F1Score', 'Accuracy', 'Precision', 'Recall'])
            dfTempCaudal = pd.DataFrame(index=[outlierDetectionMethods], columns=['F1Score', 'Accuracy', 'Precision', 'Recall'])
            
            for outMethod in outlierDetectionMethods:
                print("OUT METHOD", outMethod)
                
                sensorDict = dictOutlierMethodsBarr[outMethod] #if comp=='barreiro' else dictOutlierMethodsBarr[outMethod]
                
                allAccuracy, allPrecision, allRecall, allF1S = evaluate_multiple_outliers(usingDf, genSettings, sensorDict, '', evalIter)
                
                for s in range(len(sensors)):
                    sensorName = sensors[s]
                    
                    if(sensorName == 'Pressao'):
                        sensorAcc = allAccuracy[:,s]
                        sensorPrec = allPrecision[:,s]
                        sensorRec = allRecall[:,s]
                        sensorF1S = allF1S[:,s]
                        
                        dfTempPressao.loc[outMethod,'F1Score'] = str(round(np.mean(sensorF1S),3)) + ' \pm ' + str(round(np.std(sensorF1S),3))
                        dfTempPressao.loc[outMethod, 'Accuracy'] = str(round(np.mean(sensorAcc),3)) + ' \pm ' + str(round(np.std(sensorAcc),3))
                        dfTempPressao.loc[outMethod,'Precision'] = str(round(np.mean(sensorPrec),3)) + ' \pm ' + str(round(np.std(sensorPrec),3))
                        dfTempPressao.loc[outMethod,'Recall'] = str(round(np.mean(sensorRec),3)) + ' \pm ' + str(round(np.std(sensorRec),3))
                        
                        statName = comp + '_' + outMethod + '_' + 'pressao' + '_' + typeOut + '_' + percentOut
                        dfStats = dfStats.append({'Name': statName, 'F1Score': sensorF1S, 'Accuracy': sensorAcc, 
                                                  'Precision': sensorPrec, 'Recall': sensorRec}, ignore_index=True)
                        
                    elif(sensorName == 'Caudal'):
                        sensorAcc = allAccuracy[:,s]
                        sensorPrec = allPrecision[:,s]
                        sensorRec = allRecall[:,s]
                        sensorF1S = allF1S[:,s]
                        
                        dfTempCaudal.loc[outMethod,'F1Score'] = str(round(np.mean(sensorF1S),3)) + ' \pm ' + str(round(np.std(sensorF1S),3))
                        dfTempCaudal.loc[outMethod, 'Accuracy'] = str(round(np.mean(sensorAcc),3)) + ' \pm ' + str(round(np.std(sensorAcc),3))
                        dfTempCaudal.loc[outMethod,'Precision'] = str(round(np.mean(sensorPrec),3)) + ' \pm ' + str(round(np.std(sensorPrec),3))
                        dfTempCaudal.loc[outMethod,'Recall'] = str(round(np.mean(sensorRec),3)) + ' \pm ' + str(round(np.std(sensorRec),3))
                        
                        statName = comp + '_' + outMethod + '_' + 'caudal' + '_' + typeOut + '_' + percentOut
                        dfStats = dfStats.append({'Name': statName, 'F1Score': sensorF1S, 'Accuracy': sensorAcc, 
                                                  'Precision': sensorPrec, 'Recall': sensorRec}, ignore_index=True)
                        
                        
            captionNamePressao = comp + ' ' + 'pressao' + ' ' + typeOut + ' ' + ('2\%Out' if percentOut == '0.02' else '10\%Out')
            latexPressao = dfTempPressao.to_latex()
            tablesList.append([captionNamePressao, latexPressao])
            
            captionNameCaudal = comp + ' ' + 'caudal' + ' ' + typeOut + ' ' + ('2\%Out' if percentOut == '0.02' else '10\%Out')
            latexCaudal = dfTempCaudal.to_latex()
            tablesList.append([captionNameCaudal, latexCaudal])
    
    dfStats.to_csv('Validation/stats.csv')    
    
    return tablesList



def generate_pdf(tablesList):
        with open('Validation/validation.tex', 'w') as file:
            file.write('\\documentclass{article}\n')
            file.write('\\usepackage{booktabs}\n')
            file.write('\\begin{document}\n')
            for t in tablesList:    
                file.write('\\section{' + t[0] + '}\n')
                file.write(t[1])
            file.write('\\end{document}\n')
            
def generate_plot_best_methods(dfBarreiro, dfBeja):
    
    bestImputationMethods = ['Interpolation', 'Locf', 'Nocb', 'Random forests']
    bestOutlierDetectionMethods = ['Standard deviation', 'Inter quartile range', 'Isolation forests', 'Sax']
    
    artificialGenerationsPunctual = ['punctual,0.02', 'punctual,0.03', 'punctual,0.05', 'punctual,0.1', 'punctual,0.2']
    artificialGenerationsSequential = ['sequential,0.02', 'sequential,0.03', 'sequential,0.05', 'sequential,0.1', 'sequential,0.2']
    
    percentGenerated = [2, 3, 5, 10, 20]
    
    for comp in company:
        # chose which dataset to use and which artificialGenerationSettings to use
        usingDf = dfBarreiro if(comp=='barreiro') else dfBeja #if(comp=='beja') else dfInfraquinta
        artificialGenerations = artificialGenerationsPunctual if(comp=='barreiro') else artificialGenerationsSequential
        
        figPressao = plt.figure()
        axPressao = figPressao.add_subplot(1, 1, 1)
        yLabel = 'F1-Score'if(comp=='barreiro') else 'RMSE'
        xLabel = '% of ' 
        xLabel += 'outliers' if(comp=='barreiro') else 'missings' + ' generated artificialy'
        axPressao.set(xlabel=xLabel, ylabel=yLabel, title='Water pressure sensor')
        axPressao.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        figCaudal = plt.figure()
        axCaudal = figCaudal.add_subplot(1, 1, 1)
        axCaudal.set(xlabel=xLabel, ylabel=yLabel, title='Water flow sensor')
        axCaudal.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if(comp=='barreiro'):
            for outMethod in bestOutlierDetectionMethods:
                print("OUT METHOD", outMethod)
                
                bestScoresPressao = []
                bestScoresCaudal= []
                
                for artificialGeneration in artificialGenerations:
        
                    genSettings = generationSettings.copy()
                    
                    typeOut = artificialGeneration.split(',')[0]
                    percentOut = artificialGeneration.split(',')[1]
                    
                    genSettings.insert(0, float(percentOut))
                    genSettings.insert(1, typeOut)
                    
                    print("GENERATION SETTINGS", genSettings)
                
                    sensorDict = dictOutlierMethodsBarr[outMethod] #if comp=='barreiro' else dictOutlierMethodsBarr[outMethod]
                    
                    allAccuracy, allPrecision, allRecall, allF1S = evaluate_multiple_outliers(usingDf, genSettings, sensorDict, '', evalIter)
                    
                    for s in range(len(sensors)):
                        sensorName = sensors[s]
                        
                        if(sensorName == 'Pressao'):
                            sensorF1S = allF1S[:,s]
                            f1ScoreMean = np.mean(sensorF1S)
                            bestScoresPressao.append(f1ScoreMean)
                            
                        elif(sensorName == 'Caudal'):
                            sensorF1S = allF1S[:,s]
                            f1ScoreMean = np.mean(sensorF1S)
                            bestScoresCaudal.append(f1ScoreMean)  
                
                
                axPressao.plot(percentGenerated,bestScoresPressao, 'o--', label = outMethod)
                axCaudal.plot(percentGenerated,bestScoresCaudal, 'o--', label = outMethod)
        
        else:
            for impMethod in bestImputationMethods:
                print("IMP METHOD", impMethod)
                
                bestScoresPressao = []
                bestScoresCaudal= []
                
                for artificialGeneration in artificialGenerations:
        
                    genSettings = generationSettings.copy()
                    
                    typeNA = artificialGeneration.split(',')[0]
                    percentNA = artificialGeneration.split(',')[1]
                    
                    genSettings.insert(0, float(percentNA))
                    genSettings.insert(1, typeNA)
                    
                    print("GENERATION SETTINGS", genSettings)
                
                    sensorDict = dictImputationMethodsBarreiro[impMethod]
                    
                    allRmse, allMae, allSmape, allPercentOfImputedNA = evaluate_multiple_NA(usingDf, genSettings, sensorDict, '', evalIter)
                    
                    for s in range(len(sensors)):
                        sensorName = sensors[s]
                        
                        if(sensorName == 'Pressao'):
                            sensorRmse = allRmse[:,s]
                            rmseMean = np.mean(sensorRmse)
                            bestScoresPressao.append(rmseMean)
                            
                        elif(sensorName == 'Caudal'):
                            sensorRmse = allRmse[:,s]
                            rmseMean = np.mean(sensorRmse)
                            bestScoresCaudal.append(rmseMean)
                            
                axPressao.plot(percentGenerated,bestScoresPressao, 'o--', label = impMethod)
                axCaudal.plot(percentGenerated,bestScoresCaudal, 'o--', label = impMethod)
        
        axPressao.set_xticks(np.arange(21))
        axCaudal.set_xticks(np.arange(21))
        axPressao.legend()
        axCaudal.legend()
        figPressao.show()
        figCaudal.show()
                          
        #     else:
        #         for impMethod in bestImputationMethods:
        #             print("IMP METHOD", impMethod)
                    
        #             sensorDict = dictImputationMethodsBarreiro[impMethod]
                    
        #             allRmse, allMae, allSmape, allPercentOfImputedNA = evaluate_multiple_NA(usingDf, genSettings, sensorDict, '', evalIter)
                    
        #             for s in range(len(sensors)):
        #                 sensorName = sensors[s]
                        
        #                 if(sensorName == 'Pressao'):
        #                     sensorRmse = allRmse[:,s]
        #                     rmseMean = np.mean(sensorRmse)
        #                     bestScoresPressao.append(rmseMean)
                            
        #                 elif(sensorName == 'Caudal'):
        #                     sensorRmse = allRmse[:,s]
        #                     rmseMean = np.mean(sensorRmse)
        #                     bestScoresCaudal.append(rmseMean)

                
        
    
        # yLabel = 'F1-Score'            
        # xLabel = 'Percent of outliers generated artificialy'
        
        # x = np.array(percentGenerated)[np.newaxis, :]
        # y = np.array(bestScoresPressao)[np.newaxis, :]
        
        # print("VYUBIO", x, y)
        
        # fig, ax = plt.plot(x,y)
        
        # ax.set(xlabel=xLabel, ylabel=yLabel)#, title='About as simple as it gets, folks')
        # figName = 'Validation/best_outliers_' + comp + '_pressao.png'
        # fig.savefig(figName)
        
        # y = np.array(bestScoresCaudal)[np.newaxis, :]
        # fig, ax = plt.plot(x, y)
        # ax.set(xlabel=xLabel, ylabel=yLabel)
        # figName = 'Validation/best_outliers_' + comp + '_caudal.png'
        # fig.savefig(figName)
        
        # xLabel = 'Percent of missings generated artificialy'
        # yLabel = 'RMSE'
        
        # fig, ax = plt.subplots()
        # x = np.asarray(percentGenerated)[np.newaxis, :]
        # y = np.asarray(bestScoresPressao)[np.newaxis, :]
        
        # print("SHAPE", x.shape, y.shape)
        
        # ax.plot(x, y)
        # ax.set(xlabel=xLabel, ylabel=yLabel)#, title='About as simple as it gets, folks')
        # figName = 'Validation/best_missings_' + comp + '_pressao.png'
        # fig.savefig(figName)
        
        # y = np.asarray(bestScoresCaudal)[np.newaxis, :]
        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.set(xlabel=xLabel, ylabel=yLabel)
        # figName = 'Validation/best_missings_' + comp + '_caudal.png'
        # fig.savefig(figName)
        
    

if __name__ == "__main__":
    
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    print("COMECOU UPLOAD FILES")
    dfBarreiro, dfBeja = upload_files(numOfObservationsUsed)
    print("TERMINOU UPLOAD FILES")
    '''
    tablesListNA = generate_tables_NA(dfBarreiro, dfBeja)
    
    generate_pdf(tablesListNA)
    
 
    
    tablesListOut = generate_tables_out(dfBarreiro, dfBeja)
    
    generate_pdf(tablesListOut)
    '''
    
    generate_plot_best_methods(dfBarreiro, dfBeja)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
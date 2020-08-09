import math
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import imputation_methods, series_utils
import pandas as pd
import matplotlib.pyplot as plt
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
from datetime import datetime
import time
import os


def smapeFunction(origDataNA, predDataNA):
    return 100./len(origDataNA) * np.sum(2 * np.abs(predDataNA - origDataNA) / (np.abs(origDataNA) + np.abs(predDataNA)))


def evaluate_singular(origData, predData, nanData, index, sensor):
    origDataNA = origData.loc[index, sensor]
    predDataNA = predData.loc[index, sensor]
    
    NAindexOnPreData = predDataNA.dropna().index # only use the observations with no NA, case of Amelia
    origDataNA = origData.loc[NAindexOnPreData, sensor]
    predDataNA = predData.loc[NAindexOnPreData, sensor]
    '''
    print("ORIGDATANA", origDataNA)
    print("PREDATANA", predDataNA)
    '''
    
    #if(origDataNA.shape[0] == 0 or predDataNA.shape[0] == 0):
        #return 10, 10, 10, 0
    
    #print("INDEXES", index)
    
    #print("NANINDEXES", NAindexOnPreData)
    
    #print("predData", predData)
    
    
    
    mse = mean_squared_error(origDataNA, predDataNA)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(origDataNA, predDataNA)
    smape = smapeFunction(origDataNA, predDataNA)
        
    return mse, rmse, mae, smape
        

def evaluate_multiple(origData, imputationEvaluationSettings, sensorsDict, imputationMethodMode, evalIter): 
    
    # this allMetric is the metric for each iteration
    allMse = []
    allRmse = []
    allMae = []
    allSmape = []
    allPercentOfImputedNA = []
    
    timeSeriesToEval = series_utils.select_time_series_to_eval(origData)
    
    for i in range(evalIter):
        dfWithMissing = series_utils.generate_artificial_data(timeSeriesToEval.copy(), imputationEvaluationSettings, 'missing', i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfMissingImputed = imputation_methods.impute_missing_values(dfWithMissing.copy(), sensorsDict, imputationMethodMode, i)
            #print("DF MISSING IMPUTED", dfMissingImputed)
        sensorMse = []
        sensorRmse = []
        sensorMae = []
        sensorSmape = []
        sensorPercentOfImputedNA = []
        
        for sensor in origData.columns:            
            index = dfWithMissing.loc[dfWithMissing[sensor].isna(), :].index #indexes that will be used on the evaluation (which are the ones which had NA)
            
            percentOfImputedNA = 1 - (dfMissingImputed[sensor].isna().sum()/dfWithMissing[sensor].isna().sum())
            
            if(len(index) == 0): #if there is no NA in the sensor, continue
                continue
            
            mse, rmse, mae, smape = evaluate_singular(timeSeriesToEval, dfMissingImputed, dfWithMissing, index, sensor)
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
    
    return generate_pdf(origData.columns, np.array(allMse), np.array(allRmse), np.array(allMae), np.array(allSmape), np.array(allPercentOfImputedNA), sensorsDict, imputationEvaluationSettings)
    
def generate_pdf(sensors, allMse, allRmse, allMae, allSmape, allPercentOfImputedNA, sensorsDict, imputationEvaluationSettings):
    
    stats = ['Max', 'Min', 'Mean', 'Median', 'Std']
    pdf = FPDF()
    
    initTime = time.time()
    
    dfForCSV = None
    
    for s in range(len(sensors)):
        
        sensorName = sensors[s]
        imputationMethod = sensorsDict[sensorName][0]
        
        dfEvalTemp = pd.DataFrame(index=stats)
        sensorMse = allMse[:,s]
        sensorRmse = allRmse[:,s]
        sensorMae = allMae[:,s]
        sensorSmape = allSmape[:,s]
        sensorPercentOfImputedNA = np.mean(allPercentOfImputedNA[:,s])
        
        plt.clf()
        ''' save the boxplot  for each sensor of the MSE metric '''
        dfBoxplotMse = pd.DataFrame(sensorMse[:, np.newaxis], columns=['MSE'])
        boxplotMse = dfBoxplotMse.boxplot(column=['MSE'])
        plt.savefig('boxplotMse'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot  for each sensor of the RMSE metric '''
        dfBoxplotMse = pd.DataFrame(sensorRmse[:, np.newaxis], columns=['RMSE'])
        boxplotRmse = dfBoxplotMse.boxplot(column=['RMSE'])
        plt.savefig('boxplotRmse'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot  for each sensor of the MAE metric '''
        dfBoxplotMae = pd.DataFrame(sensorMae[:, np.newaxis], columns=['MAE'])
        boxplotMae = dfBoxplotMae.boxplot(column=['MAE'])
        plt.savefig('boxplotMae'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot for each sensor of the SMAPE metric '''
        dfBoxplotSmape = pd.DataFrame(sensorSmape[:, np.newaxis], columns=['SMAPE'])
        boxplotSmape = dfBoxplotSmape.boxplot(column=['SMAPE'])
        plt.savefig('boxplotSmape'+sensors[s]+'.png')
        plt.clf()
        
        dfEvalTemp['MSE'] = [max(sensorMse), min(sensorMse), np.mean(sensorMse), np.median(sensorMse), np.std(sensorMse)]
        
        dfEvalTemp['RMSE'] = [max(sensorRmse), min(sensorRmse), np.mean(sensorRmse), np.median(sensorRmse), np.std(sensorRmse)]
        
        dfEvalTemp['MAE'] =[max(sensorMae), min(sensorMae), np.mean(sensorMae), np.median(sensorMae), np.std(sensorMae)]
        
        dfEvalTemp['SMAPE'] = [max(sensorSmape), min(sensorSmape), np.mean(sensorSmape), np.median(sensorSmape), np.std(sensorSmape)]
    
        #print("DFEVAL", dfEval, dfEval.index, dfEval.columns)

        
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(60)
        pdf.cell(75, 10, "Evaluation of sensor " + sensorName + " using the " + imputationMethod + " method", 0, 2, 'C')
        #pdf.cell(80, 10, " ", 0, 2, 'C')
        pdf.cell(-40)
        pdf.cell(30, 10,'', 1, 0, 'C')
        pdf.cell(30, 10, 'MSE', 1, 0, 'C')
        pdf.cell(30, 10, 'RMSE', 1, 0, 'C')
        pdf.cell(30, 10, 'MAE', 1, 0, 'C')
        pdf.cell(30, 10, 'SMAPE', 1, 2, 'C')
        pdf.cell(-120)
        pdf.set_font('arial', '', 11)
        for i in range(0, len(dfEvalTemp)):
            pdf.cell(30, 10, '%s' % (dfEvalTemp.index[i]), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['MSE'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['RMSE'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['MAE'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['SMAPE'].iloc[i], 4)) + "%"), 1, 2, 'C')
            pdf.cell(-120)

        #pdf.cell(90, 10, " ", 0, 2, 'C')
        #pdf.cell(-100)
        pdf.image('boxplotMse'+sensors[s]+'.png', x = 20, y = 75, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotRmse'+sensors[s]+'.png', x = 100, y = 75, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotMae'+sensors[s]+'.png', x = 20, y = 155, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotSmape'+sensors[s]+'.png', x = 100, y = 155, w = 80, h = 80, type = '', link = '')
        pdf.set_font('arial', '', 12)
        pdf.set_xy(10, 240)
        pdf.cell(100, 10, "Parameters of the sensor " + sensorName + " are: " + str(sensorsDict[sensorName][1]), 0, 2, 'L')
        pdf.set_xy(10, 250)
        pdf.cell(100, 10, "The missing values were generated using the following parameters: " + str(imputationEvaluationSettings), 0, 2, 'L')
        pdf.set_xy(10, 260)
        pdf.cell(100, 10, str(sensorPercentOfImputedNA*100) + "% of the missing values were imputed", 0, 2, 'L')
        
        dfForCSV = pd.concat([dfForCSV, dfEvalTemp], axis=1)
    
    for file in os.listdir('.'):
        if (file.endswith('.png')):
            os.remove(file) 
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = './Reports/report_missing_'+dt_string
    pdf.output(name = filename + '.pdf', dest = 'F')
    dfForCSV.to_excel(filename + '.xlsx')
    
    endTime = time.time()
    
    print("The PDF generation process took " + str(endTime-initTime) + " seconds")
    
    return filename + '.pdf'


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

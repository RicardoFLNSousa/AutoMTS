# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:08:44 2020

@author: Ricardo
"""

import math
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import imputation_methods, series_utils, outlier_detection_methods
import pandas as pd
import matplotlib.pyplot as plt
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
from datetime import datetime
import time
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def evaluate_singular(dfWithOutliers, dfWithGeneratedOutliersIndex, detectedOutliersIndex):
    
#    
#    #print(dfWithOutliers.loc[dfWithOutliersIndex])
#    print(dfWithGeneratedOutliersIndex, detectedOutliersIndex)

    yTrue = dfWithOutliers.copy()
    
    yTrue.loc[:] = 1 # 0 is an outlier
    yTrue.loc[dfWithGeneratedOutliersIndex] = 0 # 1 is not an outlier
    yPred = dfWithOutliers.copy()
    yPred.loc[:] = 1
    yPred.loc[detectedOutliersIndex] = 0
    
    #print("LEN YPRED", yPred)
    #print("BLABLABLABALBL", len(yPred[yPred==0]))
    
    #if(len(yPred[yPred==0] == 0)): return 0,0,0,0
    
    confMatrix = confusion_matrix(yTrue, yPred)
    #print(confMatrix)
    tp = confMatrix[0][0]
    fp = confMatrix[0][1]
    fn = confMatrix[1][0]
    tn = confMatrix[1][1]
    
    accuracy = accuracy_score(yTrue, yPred)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    
    if(recall != recall): recall = 0
    if(precision != precision): precision = 0
    
    if(precision*recall == 0):#
        f1Score = 0
    else:
        f1Score = 2 * (precision * recall) / (precision + recall)
    
    #print("metrics", precision, recall, f1Score)
    
    #precision, recall, f1Score, support = precision_recall_fscore_support(yTrue, yPred, average='micro')
        
    return accuracy, precision, recall, f1Score
        

def evaluate_multiple(origData, outliersEvaluationSettings, sensorsDict, imputationMethodMode, evalIter): 
    
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
        dfWithOutliers = series_utils.generate_artificial_data(timeSeriesToEval.copy(), outliersEvaluationSettings, 'outlier', 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), sensorsDict, imputationMethodMode, 0)
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
            
            accuracy, precision, recall, f1Score = evaluate_singular(dfWithOutliers[sensor].copy(), dfGeneratedWithOutliersIndex, detectedOutliersIndex)
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
    
    return generate_pdf(origData.columns, np.array(allAccuracy), np.array(allPrecision), np.array(allRecall), np.array(allF1Score), sensorsDict, outliersEvaluationSettings)
    
def generate_pdf(sensors, allAccuracy, allPrecision, allRecall, allF1Score, sensorsDict, outliersEvaluationSettings):
    
    stats = ['Max', 'Min', 'Mean', 'Median', 'Std']
    pdf = FPDF()
    
    initTime = time.time()
    
    dfForCSV = None
    
    for s in range(len(sensors)):
        
        sensorName = sensors[s]
        outlierDetectionMethod = sensorsDict[sensorName][0]
        
        dfEvalTemp = pd.DataFrame(index=stats)
        sensorAccuracy = allAccuracy[:,s]
        sensorPrecision = allPrecision[:,s]
        sensorRecall = allRecall[:,s]
        sensorF1Score = allF1Score[:,s]
        #sensorPercentOfImputedNA = np.mean(allPercentOfImputedNA[:,s])
        
        plt.clf()
        ''' save the boxplot  for each sensor of the Accuracy metric '''
        dfBoxplotAccuracy = pd.DataFrame(sensorAccuracy[:, np.newaxis], columns=['Accuracy'])
        boxplotAccuracy = dfBoxplotAccuracy.boxplot(column=['Accuracy'])
        plt.savefig('boxplotAccuracy'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot  for each sensor of the Precision metric '''
        dfBoxplotPrecision = pd.DataFrame(sensorPrecision[:, np.newaxis], columns=['Precision'])
        boxplotPrecision = dfBoxplotPrecision.boxplot(column=['Precision'])
        plt.savefig('boxplotPrecision'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot  for each sensor of the Recall metric '''
        dfBoxplotRecall = pd.DataFrame(sensorRecall[:, np.newaxis], columns=['Recall'])
        boxplotRecall = dfBoxplotRecall.boxplot(column=['Recall'])
        plt.savefig('boxplotRecall'+sensors[s]+'.png')
        plt.clf()
        ''' save the boxplot for each sensor of the F1Score metric '''
        dfBoxplotF1Score = pd.DataFrame(sensorF1Score[:, np.newaxis], columns=['F1Score'])
        boxplotF1Score = dfBoxplotF1Score.boxplot(column=['F1Score'])
        plt.savefig('boxplotF1Score'+sensors[s]+'.png')
        plt.clf()
        
        dfEvalTemp['Accuracy'] = [max(sensorAccuracy), min(sensorAccuracy), np.mean(sensorAccuracy), np.median(sensorAccuracy), np.std(sensorAccuracy)]
        
        dfEvalTemp['Precision'] = [max(sensorPrecision), min(sensorPrecision), np.mean(sensorPrecision), np.median(sensorPrecision), np.std(sensorPrecision)]
        
        dfEvalTemp['Recall'] =[max(sensorRecall), min(sensorRecall), np.mean(sensorRecall), np.median(sensorRecall), np.std(sensorRecall)]
        
        dfEvalTemp['F1Score'] = [max(sensorF1Score), min(sensorF1Score), np.mean(sensorF1Score), np.median(sensorF1Score), np.std(sensorF1Score)]
    
        #print("DFEVAL", dfEval, dfEval.index, dfEval.columns)

        
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(60)
        pdf.cell(75, 10, "Evaluation of sensor " + sensorName + " using the " + outlierDetectionMethod + " method", 0, 2, 'C')
        #pdf.cell(80, 10, " ", 0, 2, 'C')
        pdf.cell(-40)
        pdf.cell(30, 10,'', 1, 0, 'C')
        pdf.cell(30, 10, 'ACC', 1, 0, 'C')
        pdf.cell(30, 10, 'PRE', 1, 0, 'C')
        pdf.cell(30, 10, 'REC', 1, 0, 'C')
        pdf.cell(30, 10, 'F1S', 1, 2, 'C')
        pdf.cell(-120)
        pdf.set_font('arial', '', 11)
        for i in range(0, len(dfEvalTemp)):
            pdf.cell(30, 10, '%s' % (dfEvalTemp.index[i]), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['Accuracy'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['Precision'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['Recall'].iloc[i], 4))), 1, 0, 'C')
            pdf.cell(30, 10, '%s' % (str(round(dfEvalTemp['F1Score'].iloc[i], 4))), 1, 2, 'C')
            pdf.cell(-120)

        #pdf.cell(90, 10, " ", 0, 2, 'C')
        #pdf.cell(-100)
        pdf.image('boxplotAccuracy'+sensors[s]+'.png', x = 20, y = 75, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotPrecision'+sensors[s]+'.png', x = 100, y = 75, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotRecall'+sensors[s]+'.png', x = 20, y = 155, w = 80, h = 80, type = '', link = '')
        pdf.image('boxplotF1Score'+sensors[s]+'.png', x = 100, y = 155, w = 80, h = 80, type = '', link = '')
        pdf.set_font('arial', '', 12)
        pdf.set_xy(10, 240)
        pdf.cell(100, 10, "Parameters of the sensor " + sensorName + " are: " + str(sensorsDict[sensorName][1]), 0, 2, 'L')
        pdf.set_xy(10, 250)
        pdf.cell(100, 10, "The outliers were generated using the following parameters: " + str(outliersEvaluationSettings), 0, 2, 'L')
        pdf.set_xy(10, 260)
        #pdf.cell(100, 10, str(sensorPercentOfImputedNA*100) + "% of the missing values were imputed", 0, 2, 'L')
        
        dfForCSV = pd.concat([dfForCSV, dfEvalTemp], axis=1)
    
    for file in os.listdir('.'):
        if (file.endswith('.png')):
            os.remove(file) 
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = './Reports/report_outliers_'+dt_string
    pdf.output(name = filename + '.pdf', dest = 'F')
    dfForCSV.to_excel(filename + '.xlsx')
    
    endTime = time.time()
    
    print("The PDF generation process took " + str(endTime-initTime) + " seconds")
    
    return filename + '.pdf'


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

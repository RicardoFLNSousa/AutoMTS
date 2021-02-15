# - coding: utf-8 --
'''
@info webpage for predictive analysis of metro data
@author Inês Leite and Rui Henriques
@version 1.0
'''

import pandas as pd, dash
from app import app
import gui_utils as gui
import plot_utils, series_utils, imputation_methods, outlier_detection_methods, evaluation, evaluation_outliers, bayesian, bayesian_outliers
from dash.dependencies import Input, Output, State
import dash_core_components as dcc, dash_html_components as html
import base64
import datetime 
import io
import numpy as np
import flask
from urllib.parse import quote as urlquote
import warnings
from scipy import integrate
from datetime import datetime
import time

from statsmodels.tsa.seasonal import seasonal_decompose


''' ================================ '''
''' ====== A: WEB PAGE PARAMS ====== '''
''' ================================ '''

pagetitle = 'WISDOM'
imputationMethods = ['All', 'Mean', 'Median', 'Random sample', 'Interpolation', 'Locf', 'Nocb', 'Moving average', 'Flexible moving average',
                     'Random forests', 'Expectation maximization', 'Knn', 'Mice','Amelia']

imputationMethods = ['All', 'Mean', 'Median', 'Random sample', 'Interpolation', 'Locf', 'Nocb', 'Moving average', 'Multiple moving average',
                     'Random forests', 'Expectation maximization', 'Knn', 'Mice','Amelia']

outlierDetectionMethods = ['All', 'Standard deviation', 'Inter quartile range', 'Isolation forests', 'Local outlier factor',
                           'Dbscan', 'K-means', 'Sax']

target_options = [
        #('warning',None,gui.Button.dialog),
        ('upload',None,gui.Button.upload),
        ('sensor_name',["all"],gui.Button.multidrop,["all"]), 
        ('period',['2018-01-01','2018-10-11'],gui.Button.daterange),
        ('calendar',list(gui.calendar.keys())+list(gui.week_days.keys()),gui.Button.multidrop,['all']),
        ('granularity_(minutes)','15',gui.Button.input)]
processing_parameters = [
        ('processing_mode',['missings','outliers'],gui.Button.checkbox),
        ('imputation_method_mode',['default','parametric','fully_automatic'],gui.Button.radio,'default'),
        ('imputation_mode',['univariate','multivariate'],gui.Button.checkbox),
        ('imputation_method', imputationMethods, gui.Button.unidrop, 'Mean'), 
        ('imputation_parameterization','<parameters here>',gui.Button.input),
        ('imputation_evaluation_settings','NAPercent=0.05,NAType=punctual,NAPeriod=2,NumSensor=all,SameSensorNA=False',gui.Button.input), 
        ('outliers_method_mode',['default','parametric','fully_automatic'],gui.Button.radio,'default'),
        ('outliers_mode',['point','subsequence'],gui.Button.checkbox),
        ('outliers_method', outlierDetectionMethods, gui.Button.unidrop, 'Standard deviation'), 
        ('outliers_parameterization','<parameters here>',gui.Button.input), 
        ('outliers_evaluation_settings','OutlierPercent=0.05,OutlierType=punctual,OutlierPeriod=2,NumSensor=all,SameSensorOutlier=False',gui.Button.input)]

parameters = [('Target time series',26,target_options),('Processing options',27,processing_parameters)]
charts = [('output_files_missing_values',gui.get_null_label(),gui.Button.html,True),
          ('output_files_outliers',gui.get_null_label(),gui.Button.html,True),
          ('pre_processed_output_file',gui.get_null_label(),gui.Button.html,True),
          ('statistics_report','Select parameters and run...',gui.Button.text),
          ('visualization_outliers',None,gui.Button.figure),
          ('outlier_selection', ["all"], gui.Button.multidrop,["all"]),
          ('visualization_missing_values',None,gui.Button.figure)]

layout = gui.get_layout(pagetitle,parameters,charts)

def get_states():
    return gui.get_states(target_options+processing_parameters)

def agregar(mlist):
    agregado = set()
    for entries in mlist: 
        for entry in entries: 
            agregado.add(entry) 
    return list(agregado)

''' ========================== '''
''' ====== B: DATA INFO ====== '''
''' ========================== '''

methodsParamNames = {'Mean' : ['Non Parametric Method'],
                     'Median' : ['Non Parametric Method'],
                     'Random sample' : ['Non Parametric Method'],
                     'Interpolation' : ['method'],
                     'Locf' : ['Non Parametric Method'], 
                     'Nocb' : ['Non Parametric Method'],
                     'Moving average' : ['window', 'min_periods', 'center'],
                     'Multiple moving average' : ['window', 'min_periods', 'center'],
                     'Flexible moving average' : ['window'],
                     'Random forests' : ['max_iter', 'n_estimators', 'min_samples_split', 'min_samples_leaf'],
                     'Expectation maximization' : ['loops'],
                     'Knn' : ['n_neighbors', 'weights'], 
                     'Mice' : ['m', 'defaultMethod', 'maxit'],
                     'Amelia' : ['m', 'autopri', 'max.resample']}
 


''' ============================== '''
''' ====== C: CORE BEHAVIOR ====== '''
''' ============================== '''

###################
## Aux functions ##
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

# select data from the restrictions from the Target time series panel in GUI
def get_data(states):
    
    df = pd.read_json(states['dataframe-div.children'], orient='split')

    '''A: process filter parameters'''
    
    '''A1: process sensor types filter'''
    
    '''A2: process sensors names filter'''
    sensorNames = states['sensor_name.value']
    # if there is the all option in the sensorNames list, we keep all columns
    # if there is not the all option in the sensorNames list, we just keep the columns in the list
    if('all' not in sensorNames):
        df = df[sensorNames]
        
    '''A3: process date period filter'''
    startDate, endDate = pd.to_datetime(states['period.start_date']), pd.to_datetime(states['period.end_date'])
    # restrict the dataframe time period with the startDate and endDate
    df = df.loc[startDate : endDate]
    
    '''A4: process calendar filter'''
    calendar = states['calendar.value']
    if('all' not in calendar):
        # get the day of the week monday=0, sunday=6
        weekdays = gui.get_calendar_days(calendar)
        # filter the dataframe using the index date and check if it is in the day of the weeks list
        df = df[df.index.weekday.isin(weekdays)]
    
    '''A5: process granularity'''
    
    
    #print("DF FINAL", df, df.shape)
    
    #minutes = int(states['granularidade_em_minutos.value'])
    #dias = [gui.get_calendar_days(states['calendario.value'])]
    
    
    return df #series_utils.fill_time_series(data,name,idate,fdate,minutes)

# process parameters string
def process_parameters(stringParameters):
    if(stringParameters[0] == 'No parameters available for this method'):
        return None
        
    else:
        parameters = []
        if("," in stringParameters[0]):
            splitedString = stringParameters[0].split(',')
        else:
            splitedString = stringParameters
        
        for string in splitedString:
            value = string.split('=')[1]
            if(value == 'True' or value == 'False'):
                parameters.append(True if value == 'True' else False)
            elif(value.isdigit()):
                parameters.append(int(value))
            elif('.' in value):
                parameters.append(float(value))
            else:
                parameters.append(value)
        return parameters
        

# check if the file has at least a pre-defined week period without missing values
def check_week_period(df, weeksPeriod):
    
    nanIndexes = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    
    # if there are no NaN the file is valid
    if(len(nanIndexes)==0):
        return True
    
    nanIndexes = np.insert(nanIndexes, 0, 0)
    
    # check if the difference between 2 nan indexes are equal or bigger than the week period received as parameter
    for i in range(len(nanIndexes) - 1):
        timeDif = df.index[nanIndexes[i+1]] - df.index[nanIndexes[i]]
        parsedToWeeksTimeDif = round(timeDif/ np.timedelta64(1, 'W'))
        if(parsedToWeeksTimeDif >= weeksPeriod):
            return True
    
    return False

def create_statistics_text_missings(imputationSensorsDict, timeSeriesToEval, dfWithMissing, dfMissingImputed, missingGenerationSettings):
    text = ''
    
    NAPercent = missingGenerationSettings[0]
    NAType = missingGenerationSettings[1]
    NAPeriod = missingGenerationSettings[2]
    numSensor = missingGenerationSettings[3]
    sameSensorNA = missingGenerationSettings[4]
    
    text += '######The artificial generation of missing values has the following settings######\n' 
    text += 'The percentage of artificial generated missing values is ' + str(NAPercent * 100) + '%.\n' 
    text += 'The type of missing values generated are ' + str(NAType) + ' with a period of ' + str(NAPeriod) + '.\n' 
    text += 'The sensors with artificial generated missing values were ' + str(numSensor) + ' and were generated in '
    text += 'the same ' if sameSensorNA else 'different '
    text += 'observations for each sensor.\n'
    
    text += ' \n'
    text += '######The sensors were imputed using the following methods######\n'
    for sensor in imputationSensorsDict:
        print("SENSOR", sensor)
        sensorMethod = imputationSensorsDict[sensor][0]
        sensorMethodParams = imputationSensorsDict[sensor][1]
        sensorMethodsParamNames = methodsParamNames[sensorMethod]
        index = dfWithMissing.loc[dfWithMissing[sensor].isna(), :].index
        mse, rmse,mae, smape = evaluation.evaluate_singular(timeSeriesToEval.copy(), dfMissingImputed.copy(), dfWithMissing.copy(), index, sensor)
        text += 'The sensor ' + sensor + ' was imputed with the method ' + sensorMethod + ' with a RMSE of ' + str(rmse) + ' and with the following parameters:\n '
        print("SENSORMETHODPARAMS", sensorMethodParams)
        if(sensorMethodParams is None or not sensorMethodParams):
                text += ' - No parameters available for this method \n'
        else:
            for i in range(len(sensorMethodsParamNames)):                    
                text += ' - ' + sensorMethodsParamNames[i] + ': ' + str(sensorMethodParams[i]) + '\n'
        
    return text
            

def create_statistics_text_outliers():
    
    return None
    
    
    
def integration_func(data):
    
    #print("data", data)
    
    return  integrate.simps(data) 

# detect duplicates on the data and deal with them
def detect_duplicates(df):
    
    # remove this after testing from here
    x = df.iloc[:10]
    idxList = x.index.tolist()
    idxList[5] = idxList[6]
    x.index = idxList
    # to here
    
    # detect all the duplicates indexes in the dataframe (the duplicates values are True in the duplicates list)
    duplicates = x.index.duplicated()
    
    # create a dataframe with no duplicates
    dfFinalDuplicatesImputed = x[~x.index.duplicated(keep='first')]
    # print("X", x)
    # print("Duplicate", duplicates)
    # print("DROP DUPLICATES", dfFinalDuplicatesImputed)
    
    # loop over all duplicates
    for i in range(len(duplicates)):
        if(duplicates[i]):
            # get the true index of the duplicate
            index = x.index[i]
            # turn the row which had duplicates into nan to be imputed as a missing value
            dfFinalDuplicatesImputed.loc[index, :] = np.nan
            # get all the rows with the duplicate (to be compared with the imputed dataframe dfFinalDuplicatesImputed)
            duplicateRows = x.loc[index]
            # print("INDEX", index)
            # print("VALUES", duplicateRows, duplicateRows.iloc[0], duplicateRows.columns)#duplicateRows.iloc[df.index[0],'Pressão Máx.'])
            # print("dfFinalDuplicatesImputedPREIMPUTATION", dfFinalDuplicatesImputed)
            
            for sensor in dfFinalDuplicatesImputed.columns:
                # print("SENSOR-------------", sensor)
                dfFinalDuplicatesImputed = imputation_methods.interpolation_method(dfFinalDuplicatesImputed.copy(),['linear'], sensor)
                bestRow = 0
                minDif = np.inf
                
                for row in range(len(duplicateRows)):
                    duplicatedRowValue = duplicateRows.iloc[row, duplicateRows.columns.get_loc(sensor)]
                    # print("DUPLICATE ROWS VALUE", duplicatedRowValue)
                    imputedValue = dfFinalDuplicatesImputed.loc[index,sensor]
                    # print("IMPUTED VALUE", imputedValue)
                    
                    dif = abs(duplicatedRowValue-imputedValue)
                    # print("DIF", dif)
                    if(minDif>dif): 
                        minDif = dif
                        bestRow = row
                
                dfFinalDuplicatesImputed.loc[index,sensor] = duplicateRows.iloc[bestRow, duplicateRows.columns.get_loc(sensor)]
                    
                    
                
                
            print("dfFinalDuplicatesImputedAFTERIMPUTATION", dfFinalDuplicatesImputed)
            #values
            
            
            
    
    return dfFinalDuplicatesImputed

# update the options on the dropdown menu "Outlier selection"
def update_outlier_selection_multidrop(outlierSelectionDf):
    indexes = outlierSelectionDf.index
    sensors = outlierSelectionDf.columns
    if(indexes is not None):
        # default multidrop values
        outlierSelectionValue = ['all']
        outlierSelectionOptions = [{'value': 'all', 'label': 'All'}]
        
        # for all detected outliers in the data, add for the multidrop options
        for index in indexes:
            for sensor in sensors:
            #if(index not in outlierSelectionValue):
                #selectedSensorsValue.append(sensor)
                #print("HIHI",outlierSelectionDf.loc[index,sensor])
                if(pd.isnull(outlierSelectionDf.loc[index,sensor])):
                    value = str(index) + ' & ' + sensor
                    outlierSelectionOptions.append({'value': value, 'label': value})

        return outlierSelectionValue, outlierSelectionOptions

    else: return ['all'], [{'value': 'all', 'label': 'All'}]


def remove_out_limits_observations(df):
    # get the gross errors limiters
    limitsGrossErrors = df.iloc[2,2:]
    df = df.iloc[3:]
    for sensor in limitsGrossErrors.index:
        if(not isinstance(limitsGrossErrors[sensor], float)):
            lowerLimit = float(limitsGrossErrors[sensor].split('-')[0])
            upperLimit = float(limitsGrossErrors[sensor].split('-')[1])
            df.loc[df[sensor] < lowerLimit, sensor] = np.nan
            df.loc[df[sensor] > upperLimit, sensor] = np.nan
    
    return df

##############################
## Callbacks on file upload ##
##############################
    
# upload the contents of the file and turn into a dataframe
@app.callback([Output('dataframe-div', 'children')], [Input('upload', 'contents'), Input('upload', 'filename')])
#@app.callback([Output('dataframe-div', 'children'), Output('warning', 'displayed'), Output('warning', 'message')], [Input('upload', 'contents')])

def upload_file(contents, fileName):
    
    weekPeriod = 4
    
    if(fileName is not None):
        print("--------------------------------fileName", fileName)
        company = fileName[0].split('_')[0]
        print("COMPANEY", company)
    
    if(contents is not None):
        if(company == 'barreiro'):
            content_type, content_string = contents[0].split(',')
    
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            
            # get the original columns names from the excell file
            df.columns = df.iloc[1, :].tolist() 
            
            # filter the df with the limitGrossErrors
            df = remove_out_limits_observations(df)
            
            
            #remove the initial trash rows
            #df = df.iloc[3:] 
            df = add_time_on_columns(df)
            # use the DateTime column has index, with the format YYYY-mm-dd HH:MM:SS
            df.index = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S') 
            
            #remove the initial trash columns including the original and new datetimes
            #df = df.iloc[:, 3:]
            df = df.iloc[:672, 9:11]#15]
    
            print("DF", df)
            
            #df = detect_duplicates(df.copy())
            #df = series_utils.generate_missing_rows(df.copy())
            
            isSeriesValid = check_week_period(df, weekPeriod)
            print("O FICHEIRO É VALIDO", isSeriesValid)
            if(isSeriesValid):
                jsonDf = df.to_json(orient='split')
                return [jsonDf]
                #return [jsonDf], True, 'The file is valid'
            
            else:
                return [None]
            #return [None], True, 'The file is not valid. Need at least ' + str(weekPeriod) +' weeks without NA'
        elif(company == 'beja'):
            content_type, content_string = contents[0].split(',')
    
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            
            #invert the order of the rows
            df = df.iloc[::-1]
            
            df = df.iloc[50:21800]
            
            #set the DateTime column as index
            df.index = df['DateAndTime']
            
            #filter the rows of the caudal and pressao sensors
            caudal = df.loc[df['Descritivo'] == 'Beja - Câmara Zona Alta ZMC1 Caudal']['Value']
            pressao = df.loc[df['Descritivo'] == 'Beja - Câmara Zona Alta ZMC1 Pressao']['Value']
            
            d = {'Pressao': pressao.values, 'Caudal': caudal.values}
            newDf = pd.DataFrame(d, index = pressao.index)
            
            finalDf = newDf.copy()
            finalDf = finalDf.resample('15min').mean()
            
            caudal = newDf['Caudal'].resample('15min').apply(integration_func).bfill()
            pressao = newDf['Pressao'].resample('15min').mean().bfill()
            
            finalDf['Caudal'] = caudal
            finalDf['Pressao'] = pressao
            
            finalDf = series_utils.generate_missing_rows(newDf.copy())
            
            print("finalDf", finalDf)            
            
            isSeriesValid = check_week_period(newDf, weekPeriod)
            print("O FICHEIRO É VALIDO", isSeriesValid)
            if(isSeriesValid):
                jsonDf = finalDf.to_json(orient='split')
                return [jsonDf]
            
            else:
                return [None]
            
        elif(company == 'infraquinta'):
            
            content_type, content_string = contents[0].split(',')
    
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            
            df.index = df['Row Labels']
            df = df.drop(columns=['Row Labels'])
            
            df = df.resample('60min').mean().bfill()
            
            df = df.iloc[:300]
            
            print("DF", df)
            
            isSeriesValid = check_week_period(df, weekPeriod)
            print("O FICHEIRO É VALIDO", isSeriesValid)
            if(isSeriesValid):
                jsonDf = df.to_json(orient='split')
                return [jsonDf]
            
            else:
                return [None]
        
        else:
            return [None]
            
    else: return [None]
    #else: return [None], False, 'The file is not valid'

# update the options on the dropdown menu "Sensor Names"
@app.callback([Output('sensor_name', 'value'), Output('sensor_name', 'options')],
              [Input('dataframe-div', 'children')],
              [State('sensor_name', 'value'), State('sensor_name', 'options')])
def update_sensor_name_multidrop(data, selectedSensorsValue, selectedSensorsOptions):
    if(data is not None):
        df = pd.read_json(data, orient='split')
        sensorNames = df.columns.tolist()

        # default multidrop values
        selectedSensorsValue = ['all']
        selectedSensorsOptions = [{'value': 'all', 'label': 'All'}]
        
        # for all sensors in the data, add for the multidrop options
        for sensor in sensorNames:
            if(sensor not in selectedSensorsValue):
                #selectedSensorsValue.append(sensor)
                selectedSensorsOptions.append({'value': sensor, 'label': sensor})
        
        return selectedSensorsValue, selectedSensorsOptions

    else: return ['all'], [{'value': 'all', 'label': 'All'}]
    
    
@app.callback([Output('period', 'start_date'), Output('period', 'end_date')],
              [Input('dataframe-div', 'children')])
def update_period_date_range(data):
    if(data is not None):
        df = pd.read_json(data, orient='split')
        startDate = df.index[0]
        endDate = df.index[-1]
        
        return startDate, endDate
    
    else: return '2018-10-02','2018-10-11'    
    
##############################################
## Callbacks on imputation mode check boxes ##
############################################## 

@app.callback([Output('imputation_method', 'value'),  Output('imputation_method', 'options'), Output('imputation_method', 'disabled')],
              [Input('imputation_mode', 'value'), Input('imputation_mode', 'n_clicks'), 
               Input('imputation_method_mode', 'value')])

def update_imputation_methods_options(value, n_clicks, imputationMethodMode):
    labelOptions = []
    options = []
    disable = False
    if(imputationMethodMode == 'fully_automatic'):
        options = ['Automatic']
        disable = True
    elif(not value or len(value) == 2):
        options = imputationMethods[1:]
    elif(value[0] == 'univariate'):
        options = imputationMethods[1:9]
    elif(value[0] == 'multivariate'):
        options = imputationMethods[9:]
    
    for option in options:
        labelOptions.append({'value': option, 'label': option})
        
    return options[0], labelOptions, disable

############################################
## Callbacks on outliers mode check boxes ##
############################################

@app.callback([Output('outliers_method', 'value'),  Output('outliers_method', 'options'), Output('outliers_method', 'disabled')],
              [Input('outliers_mode', 'value'), Input('outliers_mode', 'n_clicks'), 
               Input('outliers_method_mode', 'value')])

def update_outliers_methods_options(value, n_clicks, outliersMethodMode):
    labelOptions = []
    options = []
    disable = False
    if(outliersMethodMode == 'fully_automatic'):
        options = ['Automatic']
        disable = True
    elif(not value or len(value) == 2):
        options = outlierDetectionMethods[1:]
    elif(value[0] == 'point'):
        options = outlierDetectionMethods[1:]
    elif(value[0] == 'subsequence'):
        options = [outlierDetectionMethods[-1]]
    
    for option in options:
        labelOptions.append({'value': option, 'label': option})
      
    return options[0], labelOptions, disable


##############################################
## Callbacks on changing imputation methods ##
##############################################

@app.callback([Output('imputation_parameterization', 'value'), Output('imputation_parameterization', 'disabled')],
              [Input('imputation_method', 'value'), Input('imputation_method_mode', 'value')])

def update_imputation_methods_parameters(imputationMethod, mode):
    if(mode == 'default'):
        disable = True
    elif(mode == 'parametric'):
        disable = False
    elif(mode == 'fully_automatic'):
        disable = True
        return 'No parameters available for this method', disable
        
    if(imputationMethod == 'Mean'):
        return 'No parameters available for this method', disable
    elif(imputationMethod == 'Median'):
        return 'No parameters available for this method', disable
    elif(imputationMethod == 'Random sample'):
        return 'No parameters available for this method', disable
    elif(imputationMethod == 'Interpolation'):
        return 'method=linear', disable
    elif(imputationMethod == 'Locf'):
        return 'No parameters available for this method', disable
    elif(imputationMethod == 'Nocb'):
        return 'No parameters available for this method', disable
    elif(imputationMethod == 'Moving average'):
        return 'window=5,min_periods=1,center=False', disable
    elif(imputationMethod == 'Multiple moving average'):
        return 'window=5,min_periods=1,center=False', disable
    elif(imputationMethod == 'Flexible moving average'):
        return 'window=3', disable
    elif(imputationMethod == 'Random forests'):
        return 'max_iter=10,n_estimators=100,min_samples_split=2,min_samples_leaf=1', disable
    elif(imputationMethod == 'Expectation maximization'):
        return 'loops=50', disable
    elif(imputationMethod == 'Knn'):
        return 'n_neighbors=5,weights=uniform', disable
    elif(imputationMethod == 'Mice'):
        return 'm=5,defaultMethod=pmm,maxit=5,norm=False', disable
    elif(imputationMethod == 'Amelia'):
        return 'm=5,autopri=0.05,max.resample=100,norm=False', disable
    else:
        return '<parameters here>', disable
            
    
######################################################
## Callbacks on changing outliers detection methods ##
######################################################

@app.callback([Output('outliers_parameterization', 'value'), Output('outliers_parameterization', 'disabled')],
              [Input('outliers_method', 'value'), Input('outliers_method_mode', 'value')])

def update_outlier_detection_methods_parameters(outlierDetectionMode, mode):
    if(mode == 'default'):
        disable = True
    elif(mode == 'parametric'):
        disable = False
    elif(mode == 'fully_automatic'):
        disable = True
        return 'No parameters available for this method', disable
    
    if(outlierDetectionMode == 'Standard deviation'):
        return 'No parameters available for this method', disable
    elif(outlierDetectionMode == 'Inter quartile range'):
        return 'No parameters available for this method', disable
    elif(outlierDetectionMode == 'Isolation forests'):
        return 'n_estimators=100,max_features=1', disable
    elif(outlierDetectionMode == 'Local outlier factor'):
        return 'n_neighbors=20,metric=minkowski', disable
    elif(outlierDetectionMode == 'Dbscan'):
        return 'eps=0.5,min_samples=5,metric=euclidean', disable
    elif(outlierDetectionMode == 'K-means'):
        return 'No parameters available for this method', disable
    elif(outlierDetectionMode == 'Sax'):
        return 'strategy=uniform', disable
    else:
        return '<parameters here>', disable
    
######################################################
################### LINK PDF #########################
######################################################

@app.server.route('/tdownloads/<path:fname>')
def serve_traffic_static(fname):
    return flask.send_from_directory('.',fname)#.replace('/','\\')

#################################
## Callbacks on button pressed ##
#################################

@app.callback([Output('output_files_missing_values','children'), Output('output_files_outliers','children'),Output('statistics_report','value'),
               Output('visualization_missing_values','figure'),Output('visualization_outliers', 'figure'), 
               Output('outlier_selection', 'value'), Output('outlier_selection', 'options'),
               Output('missing-sensordict-div', 'children'), Output('outlier-sensordict-div', 'children')], 
              [Input('main_button','n_clicks')],
              [State('dataframe-div', 'children'), State('period', 'start_date'), State('period', 'end_date'),
               State('sensor_name', 'value'), State('calendar', 'value'), State('imputation_method', 'value'),
               State('outliers_method', 'value'), State('imputation_parameterization', 'value'), 
               State('outliers_parameterization', 'value'), State('imputation_evaluation_settings', 'value'),
               State('imputation_method_mode', 'value'), State('outliers_evaluation_settings', 'value'),
               State('outliers_method_mode', 'value'), State('processing_mode', 'value')])

def update_charts(inp,*args):
    outlierSelectionValue = ['all']
    outlierSelectionOptions = [{'value': 'all', 'label': 'All'}]
    imputationSensorsDict = None
    outliersSensorsDict = None
    if inp is None: 
        nullplot = plot_utils.get_null_plot()
        return None, None, ["No information to show"], nullplot, nullplot, outlierSelectionValue, outlierSelectionOptions, imputationSensorsDict, outliersSensorsDict
    states = dash.callback_context.states
    
    nullplot = plot_utils.get_null_plot()
    figOutliers = nullplot
    figMissing = nullplot
    
    childreOutliers = None
    childrenMissing = None
    
    evalIter = 1 #number of iterations during the multiple evaluation
    
    processingMode = states['processing_mode.value']
    
    '''A: filter series with parameters from GUI'''
    originalDf = get_data(states)
    
    '''Select the data in series with most periods without missing value used to evaluate'''
    timeSeriesToEval = series_utils.select_time_series_to_eval(originalDf.copy())
    #pd.set_option('display.max_rows', df.shape[0]+1)
    print("TIMESERIESTOEVAL", timeSeriesToEval)
    statisticsInfoPanelMissings = ''
    
    if('outliers' in  processingMode or len(processingMode)== 0):
        '''GENERATE ARTIFICIAL OUTLIERS VALUES JUST FOR TESTING -> REMOVE WHEN IT IS DONE'''
        outliersGenerationSettings = process_parameters([states['outliers_evaluation_settings.value']]) 
        dfWithOutliers = series_utils.generate_artificial_data(timeSeriesToEval.copy(), outliersGenerationSettings, 'outlier', 0)
        #print("dfWithOutliers", dfWithOutliers)
        
    #    '''detect outliers using the selected outlier detection method'''
        outliersMethodMode = states['outliers_method_mode.value']
        outliersMethod = states['outliers_method.value']
        
        '''HyperOpt'''
        if(outliersMethodMode == 'fully_automatic'):
            outliersSensorsDict = bayesian_outliers.hyperparameter_tuning_bayesian(timeSeriesToEval.copy(), outliersGenerationSettings, outlierDetectionMethods[1:])#outlierDetectionMethods[1:]) #TODO meter todos os metodos outlierDetectionMethods[1:]
            print("TODO FULLY AUTOMATIC")
        else:
            '''Non HyperOpt'''
            stringOutliersParameters = states['outliers_parameterization.value']
            outliersParameters = process_parameters([stringOutliersParameters])
            outliersSensorsDict = dict.fromkeys(originalDf.columns, [outliersMethod, outliersParameters])
            
        print("-------------OutliersSensorsDict---------------", outliersSensorsDict)
        
        '''evaluate the methods with 30 runs'''
        fileNameOutliers = evaluation_outliers.evaluate_multiple(timeSeriesToEval.copy(), outliersGenerationSettings, outliersSensorsDict, outliersMethodMode, evalIter)
        
        # USADO APENAS NOS TESTES. A DFWITHOUTLIERS FOI GERADA POR NOS
        #dfDetectedOutliers, dfWithDetectedOutliersToNan = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), outliersSensorsDict, outliersMethodMode, 0)
        
        dfDetectedOutliers, dfWithDetectedOutliersToNan = outlier_detection_methods.detect_outliers(originalDf.copy(), outliersSensorsDict, outliersMethodMode, 0)
        
        # example of the evaluate_singular
        '''
        for sensor in dfWithOutliers.columns:
            dfGeneratedWithOutliersIndex = timeSeriesToEval[dfWithOutliers[sensor] != timeSeriesToEval[sensor]].index
            print("SENSOR", sensor)
            detectedOutliersIndex = dfWithDetectedOutliersToNan.loc[dfWithDetectedOutliersToNan[sensor].isna(), :].index #indexes that will be used on the evaluation (which are the ones that were detected as outliers)
            accuracy, precision, recall, f1Score = evaluation_outliers.evaluate_singular(dfWithOutliers[sensor].copy(), dfGeneratedWithOutliersIndex, detectedOutliersIndex)
            print("SCORES", accuracy, precision, recall, f1Score)
        '''
        #print("DETECTED OUTLIERS", dfDetectedOutliers, dfWithDetectedOutliersToNan)
        
    #    '''
    #    outlierDetectionMethod = states['outliers_method.value']
    #    stringOutlierParameters = states['outliers_parameterization.value']
    #    outlierParameters = process_parameters(stringOutlierParameters)
    #    dfOutliers = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), outlierDetectionMethod, outlierParameters)
    #    print("dfOutliers", dfOutliers)
    #    '''
        figOutliers = plot_utils.get_series_plot(dfWithOutliers,'Outliers Detection', dfDetectedOutliers, 'outliers')
        childreOutliers = [dcc.Link(fileNameOutliers[10:],href='/tdownloads/'+ fileNameOutliers,refresh=False, target="_blank"),html.Br()]
        
        # mask with the detected outliers at true
        mask = dfWithDetectedOutliersToNan.isna().values
        # dataframe with the detected outliers only
        outlierSelectionDf = dfWithDetectedOutliersToNan[mask].drop_duplicates()
        print("OUTLEIRSELECTIONDF", outlierSelectionDf)

        # get the values to change on the multidrop of outlierselection
        outlierSelectionValue, outlierSelectionOptions = update_outlier_selection_multidrop(outlierSelectionDf)
    
    if('missings' in processingMode or len(processingMode) == 0):
        
        print("MISSINGS ATIVO")
        
        '''GENERATE ARTIFICIAL MISSING VALUES JUST FOR TESTING -> REMOVE WHEN IT IS DONE'''
        missingGenerationSettings = process_parameters([states['imputation_evaluation_settings.value']])    
        print("MISSING GENERATION SETTINGS",missingGenerationSettings)
        dfWithMissing = series_utils.generate_artificial_data(timeSeriesToEval.copy(), missingGenerationSettings, 'missing', 0)
        print("DFWITHMISSING", dfWithMissing) 
        
        '''B: Preprocess time series'''
        '''impute missing values using the selected imputation method'''
        imputationMethodMode = states['imputation_method_mode.value']
        imputationMethod = states['imputation_method.value']
        
        '''HyperOpt'''
        if(imputationMethodMode == 'fully_automatic'):
            imputationSensorsDict = bayesian.hyperparameter_tuning_bayesian(timeSeriesToEval.copy(), missingGenerationSettings, imputationMethods[1:]) #TODO meter todos os metodos imputationMethods[1:]
    
        else:
            '''Non HyperOpt'''
            stringImputationParameters = states['imputation_parameterization.value']
            imputationParameters = process_parameters([stringImputationParameters])
            imputationSensorsDict = dict.fromkeys(originalDf.columns, [imputationMethod, imputationParameters])
            
        #print("-------------ImputationSensorsDict---------------", imputationSensorsDict)
            
        '''evaluate the methods with 30 runs'''
        fileNameMissing = evaluation.evaluate_multiple(timeSeriesToEval.copy(), missingGenerationSettings, imputationSensorsDict, imputationMethodMode, evalIter)
    
        print("-------------------------FINAAAAAAAAAAAAAAAAAAAAL-----------------------")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # USADO PARA TESTES. DFWITHMISSING FOI GERADA POR NOS
            #dfMissingImputed = imputation_methods.impute_missing_values(dfWithMissing.copy(), imputationSensorsDict, imputationMethodMode, 0)
            
            dfMissingImputed = imputation_methods.impute_missing_values(originalDf.copy(), imputationSensorsDict, imputationMethodMode, 0)
            
       #pd.set_option('display.max_rows', df.shape[0]+1)
        #print(dfMissingImputed)
        
        figMissing = plot_utils.get_series_plot(dfMissingImputed,'Missings Imputation', dfWithMissing, 'missings')
        childrenMissing = [dcc.Link(fileNameMissing[10:],href='/tdownloads/'+ fileNameMissing,refresh=False, target="_blank"),html.Br()]
    
        statisticsInfoPanelMissings = create_statistics_text_missings(imputationSensorsDict, timeSeriesToEval, dfWithMissing, dfMissingImputed, missingGenerationSettings)
#    
    
#       
    
#    
#    '''C: Plot time series'''
#    
    
    
#    nullplot = plot_utils.get_null_plot()
#
#    '''D: Plot statistics'''
#
#    
#    #return fig, corr
#    return childrenMissing, [statisticsInfoPanelMissings], figMissing, figOutliers


    return childrenMissing, childreOutliers, statisticsInfoPanelMissings, figMissing, figOutliers, outlierSelectionValue, outlierSelectionOptions, imputationSensorsDict, outliersSensorsDict


@app.callback([Output('pre_processed_output_file','children')], 
              [Input('final_button','n_clicks')],
              [State('missing-sensordict-div', 'children'), State('outlier-sensordict-div', 'children'), 
               State('dataframe-div', 'children'),
               State('period', 'start_date'), State('period', 'end_date'),
               State('sensor_name', 'value'), State('calendar', 'value'), State('imputation_method', 'value'),
               State('outliers_method', 'value'), State('imputation_parameterization', 'value'), 
               State('outliers_parameterization', 'value'), State('imputation_evaluation_settings', 'value'),
               State('imputation_method_mode', 'value'), State('outliers_evaluation_settings', 'value'),
               State('outliers_method_mode', 'value'), State('processing_mode', 'value'),
               State('outlier_selection', 'value'), State('outlier_selection', 'options')])

def remove_outliers_and_generate(inp,*args):
    if inp is None: 
        return [None]
    
    states = dash.callback_context.states
    originalDf = get_data(states)
    print("ORIGINAL DF", originalDf)

    
    outlierSelectionValue = states['outlier_selection.value']
    print("OUTLIER SELECTION VALUE", outlierSelectionValue)
    
    outlierSelectionOptions = states['outlier_selection.options']
    print("OUTLIER SELECTION OPTIONS", outlierSelectionOptions)
    
    imputationSensorsDict = states['missing-sensordict-div.children']
    print("IMPUTATION SENSOR DICT", imputationSensorsDict)
    
    outlierSensorsDict  = states['outlier-sensordict-div.children']
    print("OUTLIER SENSOR DICT", outlierSensorsDict)
    
    if(outlierSensorsDict is not None):
        if('all' in outlierSelectionValue):
            outlierSelectionOptions = outlierSelectionOptions[1:]
            for i in range(len(outlierSelectionOptions)):
                print("value", outlierSelectionOptions[i]['value'])
                value = outlierSelectionOptions[i]['value']
                split = value.split(' & ')
                print("split", split)
                index  = split[0]
                print("index", index)
                dateTimeIndex = pd.to_datetime(index)
                print("datetimeindex", dateTimeIndex)
                sensor = split[1]
                print("sensor",sensor)
                print(originalDf.loc[dateTimeIndex, sensor])
                originalDf.loc[dateTimeIndex, sensor] = np.nan
                print(originalDf.loc[dateTimeIndex, sensor])
        else:
            for i in range(len(outlierSelectionValue)):
                value = outlierSelectionValue[i]
                print("value")
                split = value.split(' & ')
                print("split", split)
                index  = split[0]
                print("index", index)
                dateTimeIndex = pd.to_datetime(index)
                print("datetimeindex", dateTimeIndex)
                sensor = split[1]
                print("sensor",sensor)
                print(originalDf.loc[dateTimeIndex, sensor])
                originalDf.loc[dateTimeIndex, sensor] = np.nan
                print(originalDf.loc[dateTimeIndex, sensor])

    print(originalDf)        
    if(imputationSensorsDict is not None):
        originalDf = imputation_methods.impute_missing_values(originalDf.copy(), imputationSensorsDict, '', 0)
    
    csv = originalDf.to_csv()
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = './Reports/processed_file_'+dt_string+'.xlsx'
    originalDf.to_excel(filename)
    link = [dcc.Link(filename[10:],href='/tdownloads/'+ filename,refresh=False, target="_blank"),html.Br()]
    
    return [link]


''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()

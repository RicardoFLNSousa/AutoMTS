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

from statsmodels.tsa.seasonal import seasonal_decompose


''' ================================ '''
''' ====== A: WEB PAGE PARAMS ====== '''
''' ================================ '''

pagetitle = 'WISDOM'
imputationMethods = ['All', 'Mean', 'Median', 'Random sample', 'Interpolation', 'Locf', 'Nocb', 'Moving average', 'Flexible moving average',
                     'Random forests', 'Expectation maximization', 'Knn', 'Mice','Amelia']

imputationMethods = ['All', 'Mean', 'Median', 'Random sample', 'Interpolation', 'Locf', 'Nocb', 'Moving average', 
                     'Random forests', 'Expectation maximization', 'Knn', 'Mice','Amelia']

outlierDetectionMethods = ['All', 'Standard deviation', 'Inter quartile range', 'Isolation forests', 'Local outlier factor',
                           'Dbscan', 'K-means', 'Sax']

target_options = [
        #('warning',None,gui.Button.dialog),
        ('upload',None,gui.Button.upload),
        ('sensor_type',["all"],gui.Button.multidrop,["all"]), 
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
        ('imputation_evaluation_metrics',["all", "mse", 'rmse', 'mae', 'smape'],gui.Button.multidrop,["all", "mse", 'rmse', 'mae', 'smape']),
        ('outliers_method_mode',['default','parametric','fully_automatic'],gui.Button.radio,'default'),
        ('outliers_mode',['point','subsequence'],gui.Button.checkbox),
        ('outliers_method', outlierDetectionMethods, gui.Button.unidrop, 'Standard deviation'), 
        ('outliers_parameterization','<parameters here>',gui.Button.input), 
        ('outliers_evaluation_settings','OutlierPercent=0.05,OutlierType=punctual,OutlierPeriod=2,NumSensor=all,SameSensorOutlier=False',gui.Button.input)]

parameters = [('Target time series',28,target_options),('Processing options',28,processing_parameters)]
charts = [('output_files_missing_values',gui.get_null_label(),gui.Button.html,True),
          ('output_files_outliers',gui.get_null_label(),gui.Button.html,True),
          ('statistics_report','Select parameters and run...',gui.Button.text),
          ('visualization_missing_values',None,gui.Button.figure), ('visualization_outliers',None,gui.Button.figure)]

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
        mse, rmse,mae, smape = evaluation.evaluate_singular(timeSeriesToEval.copy(), dfMissingImputed.copy(), dfWithMissing.copy(), index)
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
            
            #remove the initial trash rows
            df = df.iloc[3:] 
            df = add_time_on_columns(df)
            # use the DateTime column has index, with the format YYYY-mm-dd HH:MM:SS
            df.index = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S') 
            
            #remove the initial trash columns including the original and new datetimes
            #df = df.iloc[:, 3:]
            df = df.iloc[:672, 9:15]
            print("DF", df)
            
            
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
            
            finalDf
            
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
               Output('visualization_missing_values','figure'),Output('visualization_outliers', 'figure')], 
              [Input('button','n_clicks')],
              [State('dataframe-div', 'children'), State('period', 'start_date'), State('period', 'end_date'),
               State('sensor_name', 'value'), State('calendar', 'value'), State('imputation_method', 'value'),
               State('outliers_method', 'value'), State('imputation_parameterization', 'value'), 
               State('outliers_parameterization', 'value'), State('imputation_evaluation_settings', 'value'),
               State('imputation_method_mode', 'value'), State('outliers_evaluation_settings', 'value'),
               State('outliers_method_mode', 'value'), State('processing_mode', 'value')])

def update_charts(inp,*args):
    if inp is None: 
        nullplot = plot_utils.get_null_plot()
        return None, None, ["No information to show"], nullplot, nullplot
    states = dash.callback_context.states
    
    nullplot = plot_utils.get_null_plot()
    figOutliers = nullplot
    figMissing = nullplot
    
    childreOutliers = None
    childrenMissing = None
    
    evalIter = 1 #number of iterations during the multiple evaluation
    
    processingMode = states['processing_mode.value']
    
    '''A: filter series with parameters from GUI'''
    df = get_data(states)
    
    '''Select the data in series with most periods without missing value used to evaluate'''
    timeSeriesToEval = series_utils.select_time_series_to_eval(df)
    #pd.set_option('display.max_rows', df.shape[0]+1)
    print("TIMESERIESTOEVAL", timeSeriesToEval)
    
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
            imputationSensorsDict = dict.fromkeys(df.columns, [imputationMethod, imputationParameters])
            
        #print("-------------ImputationSensorsDict---------------", imputationSensorsDict)
            
        '''evaluate the methods with 30 runs'''
        fileNameMissing = evaluation.evaluate_multiple(timeSeriesToEval.copy(), missingGenerationSettings, imputationSensorsDict, imputationMethodMode, evalIter)
    
        print("-------------------------FINAAAAAAAAAAAAAAAAAAAAL-----------------------")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfMissingImputed = imputation_methods.impute_missing_values(dfWithMissing.copy(), imputationSensorsDict, imputationMethodMode, 0)
        #pd.set_option('display.max_rows', df.shape[0]+1)
        #print(dfMissingImputed)
        
        figMissing = plot_utils.get_series_plot(dfMissingImputed,'Missings Imputation', dfWithMissing, 'missings')
        childrenMissing = [dcc.Link(fileNameMissing[10:],href='/tdownloads/'+ fileNameMissing,refresh=False, target="_blank"),html.Br()]
    
#    
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
            outliersSensorsDict = bayesian_outliers.hyperparameter_tuning_bayesian(timeSeriesToEval.copy(), outliersGenerationSettings, outlierDetectionMethods[1:]) #TODO meter todos os metodos outlierDetectionMethods[1:]
            print("TODO FULLY AUTOMATIC")
        else:
            '''Non HyperOpt'''
            stringOutliersParameters = states['outliers_parameterization.value']
            outliersParameters = process_parameters([stringOutliersParameters])
            outliersSensorsDict = dict.fromkeys(df.columns, [outliersMethod, outliersParameters])
            
        print("-------------OutliersSensorsDict---------------", outliersSensorsDict)
        
        '''evaluate the methods with 30 runs'''
        fileNameOutliers = evaluation_outliers.evaluate_multiple(timeSeriesToEval.copy(), outliersGenerationSettings, outliersSensorsDict, outliersMethodMode, evalIter)
        
        dfDetectedOutliers, dfWithDetectedOutliersToNan = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), outliersSensorsDict, outliersMethodMode, 0)
        
        
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
    
#    
#    statisticsInfoPanelMissings = create_statistics_text_missings(imputationSensorsDict, timeSeriesToEval, dfWithMissing, dfMissingImputed, missingGenerationSettings)
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

    return childrenMissing, childreOutliers, ["No information to show"], figMissing, figOutliers


''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()

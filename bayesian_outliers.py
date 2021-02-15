from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval, STATUS_FAIL
from hyperopt.pyll import scope
import numpy as np
import series_utils, outlier_detection_methods, evaluation_outliers
import warnings

randomState = 0


paramHyperOptIsolationForests= {'n_estimators': hp.choice("n_estimators", range(100,101,1)),
                                'max_features': hp.choice("max_features", np.arange(0.5,1,.1))}

paramHyperOptLocalOutlierFactor = {'n_neighbors' : hp.choice('n_neighbors', range(10,35,5)),
                                   'metric': hp.choice('metric', ['minkowski', 'hamming', 'chebyshev', 'euclidean'])}

paramHyperOptDbscan = {'eps' : hp.choice('eps',  np.arange(0.5,5,1.5)),
                       'min_samples': hp.choice("min_samples", range(3,9,2)),
                       'metric': hp.choice('metric', ['hamming', 'chebyshev', 'euclidean'])}

paramHyperOptKmeans = {'n_clusters' : hp.choice("n_clusters", range(6,12,2))}



#Funcão que retorna os parametros consoante o score. Como estou a utilizar o fmin, o a minha loss function é 1-score
def hyperopt(paramHyperopt, df, dfWithOutliers, numEval, sensor, method):

	#funcão que vai ser minimizada
    def objective_function(params):
        #print("PARAMS", params)
        # Verify if a set of parameters was already tested and if they were return STATUS_FAIL
        if len(trials.trials)>1:
            for x in trials.trials[:-1]:
                space_point_index = dict([(key,value[0]) for key,value in x['misc']['vals'].items() if len(value)>0])
                if params == space_eval(paramHyperopt, space_point_index):
                    loss = x['result']['loss']
                    return {'loss': loss, 'status': STATUS_FAIL}
            
        if(method == 'Isolation forests'):
            #print("ENTRIE ISOLATIONF OREST")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #print("DF WITH OUTLIERS ISOLATION FORESTS", dfWithOutliers)
                reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.isolation_forest(dfWithOutliers.copy(), [params['n_estimators'], params['max_features'], 0], sensor)
            #print("SAI ISOLATION FORESTS")
            
        elif(method == 'Local outlier factor'):
            reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.local_outlier_factor(dfWithOutliers.copy(), [params['n_neighbors'], params['metric']], sensor)

        elif(method == 'Dbscan'):
           reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.dbscan(dfWithOutliers.copy(), [params['eps'], params['min_samples'], params['metric']], sensor)

        detectedOutliersIndex = reconstructedDataFinalNan.loc[reconstructedDataFinalNan.isna()].index
        
        dfGeneratedWithOutliersIndex = df[dfWithOutliers[sensor] != df[sensor]].index
    
        accuracy, precision, recall, f1Score = evaluation_outliers.evaluate_singular(dfWithOutliers[sensor].copy(), dfGeneratedWithOutliersIndex, detectedOutliersIndex)
        
        print("f1Score hyperopt", method, accuracy, precision, recall, f1Score)
        return {'loss': f1Score, 'status': STATUS_OK}
        #return 1-rmse
        
    
    trials = Trials()
	#melhores parametros
    bestParam = fmin(objective_function,
					 paramHyperopt,
					 algo=tpe.suggest,
					 max_evals=numEval,
					 trials=trials,
					 rstate=np.random.RandomState(randomState),
					 verbose=1
					 )

    return bestParam


def hyperparameter_tuning_bayesian(df, outliersGenerationSettings, methods):
    
    paramHyperopt = None
    dfWithOutliers = series_utils.generate_artificial_data(df.copy(), outliersGenerationSettings, 'outlier', randomState)
    MAX_EVALS = 100
    #MAX_EVALS = 10
    sensors = df.columns
    print("SENSORES", sensors)
    sensorsDict = dict.fromkeys(sensors, None)
    
    for sensor in sensors:
        print("-------SENSOR------", sensor)
        fakeSensorDict = dict.fromkeys([sensor], None)
        bestMethodF1Score = 0
        bestMethodEval = methods[0]
        bestMethodParameters = []
        
        for method in methods:
            print("METHOD", method)
            parameters = []
            
            '''Conditions for parametrics methods'''
            if(method == 'Isolation forests'):
                paramHyperopt = paramHyperOptIsolationForests
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithOutliers.copy(), MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                n_estimators = bestParamConverted['n_estimators']
                max_features = bestParamConverted['max_features']
                parameters = [n_estimators, max_features]
            
            elif(method == 'Local outlier factor'):
                paramHyperopt = paramHyperOptLocalOutlierFactor
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithOutliers.copy(), MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                n_neighbors = bestParamConverted['n_neighbors']
                metric = bestParamConverted['metric']
                parameters = [n_neighbors, metric]
                print("PARAMETERS DENTRO DO ELIF", parameters)
                
            elif(method == 'Dbscan'):
                paramHyperopt = paramHyperOptDbscan
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithOutliers.copy(), MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                eps = bestParamConverted['eps']
                min_samples = bestParamConverted['min_samples']
                metric = bestParamConverted['metric']
                parameters = [eps, min_samples, metric]


            fakeSensorDict[sensor] = [method, parameters]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reconstructedDataFinal, reconstructedDataFinalNan = outlier_detection_methods.detect_outliers(dfWithOutliers.copy(), fakeSensorDict, 'None', randomState)
            
            #reconstructedDataFinal = reconstructedDataFinal[sensor]
            #reconstructedDataFinalNan = reconstructedDataFinalNan[sensor]
            
            detectedOutliersIndex = reconstructedDataFinalNan.loc[reconstructedDataFinalNan[sensor].isna(), :].index
            
            dfGeneratedWithOutliersIndex = df[dfWithOutliers[sensor] != df[sensor]].index
            
            #print("_--------------------_", detectedOutliersIndex, dfGeneratedWithOutliersIndex)
            
            accuracy, precision, recall, f1Score = evaluation_outliers.evaluate_singular(dfWithOutliers[sensor].copy(), dfGeneratedWithOutliersIndex, detectedOutliersIndex)
            
            if(f1Score > bestMethodF1Score):
                bestMethodF1Score = f1Score
                bestMethodEval = method
                bestMethodParameters = parameters
                
            sensorsDict[sensor] = [bestMethodEval, bestMethodParameters]
            
    return sensorsDict
    


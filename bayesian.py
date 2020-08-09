from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval, STATUS_FAIL
from hyperopt.pyll import scope
import numpy as np
import series_utils, imputation_methods, evaluation
import warnings

randomState = 0
'''
paramHyperOptMovingAverage = {'window' : hp.choice('window', range(5,6,1)),
                              'min_periods': hp.choice('min_periods', range(1,2,1)),
                              'center': hp.choice("center", [True, False])} 

paramHyperOptRandomForests = {'max_iter' : hp.choice('max_iter', range(10,11,1)),
                              'n_estimators' : hp.choice('n_estimators', range(100,101,1)),
                              'min_samples_split' : hp.choice('min_samples_split',range(2,3,1)),
                              'min_samples_leaf' : hp.choice('min_samples_leaf', range(1,2,1))}


paramHyperOptEM = {'loops' : hp.choice('loops', range(50,51,1))}

paramHyperOptKnn = {'n_neighbors' : hp.choice('n_neighbors', range(5,6,1)),
                    'weights' : hp.choice('weights', ['uniform'])}

paramHyperOptMice = {'m' : hp.choice('m', range(5,6,1)),
                    'defaultMethod' : hp.choice('weights', ['pmm']),
                    'maxit' : hp.choice('maxit', range(5,6,1))}

paramHyperOptAmelia = {'m' : hp.choice('m', range(5,6,1)),
                       'autopri' : hp.choice('autpri', [0.05]),
                       'max.resample' : hp.choice('max.resample', range(100,101,1))}

'''

paramHyperOptInterpolation = {'method': hp.choice("method", ['linear', 'polynomial'])}

paramHyperOptMovingAverage = {'window' : hp.choice('window', range(4,12,2)),
                              'min_periods': hp.choice('min_periods', range(1,5,1)),
                              'center': hp.choice("center", [True, False])}

paramHyperOptFlexMovingAverage = {'window' : hp.choice('window', range(3,11,2))}

paramHyperOptRandomForests = {'max_iter' : hp.choice('max_iter', range(8,15,1)),
                              'n_estimators' : hp.choice('n_estimators', range(100,120,2)),
                              'min_samples_split' : hp.choice('min_samples_split',range(2,5,1)),
                              'min_samples_leaf' : hp.choice('min_samples_leaf', range(1,5,1))}


paramHyperOptEM = {'loops' : hp.choice('loops', range(30,110,10))}

paramHyperOptKnn = {'n_neighbors' : hp.choice('n_neighbors', range(1,21,2)),
                    'weights' : hp.choice('weights', ['uniform', 'distance'])}

paramHyperOptMice = {'m' : hp.choice('m', range(5,11,1)),
                    'defaultMethod' : hp.choice('weights', ['pmm']),
                    'maxit' : hp.choice('maxit', range(5,10,1)),
                    'norm': hp.choice("center", [True, False])}

paramHyperOptAmelia = {'m' : hp.choice('m', range(5,11,1)),
                       'autopri' : hp.choice('autpri', [0.05, 0.1, 0.15, 0.2]),
                       'max.resample' : hp.choice('max.resample', range(100,120,2)),
                       'norm': hp.choice("center", [True, False])}


#Funcão que retorna os parametros consoante o score. Como estou a utilizar o fmin, o a minha loss function é 1-score
def hyperopt(paramHyperopt, df, dfWithMissing, indexNA, numEval, sensor, method):

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
            
        if(method == 'Interpolation'):   
            dfImputed = imputation_methods.interpolation_method(dfWithMissing, [params['method']], sensor)
            
        elif(method == 'Moving average'):
            #dfImputed = imputation_methods.moving_average(dfWithMissing, [params['window'], params['min_periods'], params['center']], sensor)
            if(params['window'] <= params['min_periods']):
                return {'loss': 1000, 'status': STATUS_FAIL}
            
            dfImputed = imputation_methods.moving_average(dfWithMissing, [params['window'], params['min_periods'], params['center']], sensor)
            #print("DFIMPUTED", dfImputed)
        elif(method == 'Flexible moving average'):
            #dfImputed = imputation_methods.moving_average(dfWithMissing, [params['window'], params['min_periods'], params['center']], sensor)
            dfImputed = imputation_methods.flexible_moving_average(dfWithMissing, [params['window']], sensor)
            #print("DFIMPUTED", dfImputed)
        elif(method == 'Random forests'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfImputed = imputation_methods.random_forests(dfWithMissing, [params['max_iter'], params['n_estimators'], 
                                                                          params['min_samples_split'], params['min_samples_leaf'], 0], sensor)
        elif(method == 'Expectation maximization'):
            dfImputed = imputation_methods.mtsdi(dfWithMissing, [params['loops']], sensor)
            
        elif(method == 'Knn'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfImputed = imputation_methods.knn(dfWithMissing, [params['n_neighbors'], params['weights']], sensor)   
        
        elif(method == 'Mice'):
            dfImputed = imputation_methods.mice(dfWithMissing, [params['m'], params['defaultMethod'], params['maxit'], params['norm']], sensor)
        
        elif(method == 'Amelia'):
            dfImputed = imputation_methods.amelia(dfWithMissing, [params['m'], params['autopri'], params['max.resample'], params['norm']], sensor)
        

        mse, rmse, mae, smape = evaluation.evaluate_singular(df.copy(), dfImputed, dfWithMissing, indexNA)
        
        #print("RMSE", rmse)
        return {'loss': rmse, 'status': STATUS_OK}
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


def hyperparameter_tuning_bayesian(df, imputationEvaluationSettings, methods):
    
    paramHyperopt = None
    dfWithMissing = series_utils.generate_artificial_data(df.copy(), imputationEvaluationSettings, 'missing', randomState)
    MAX_EVALS = 100
    #MAX_EVALS = 10
    sensors = df.columns
    #print("SENSORES", sensors)
    sensorsDict = dict.fromkeys(sensors, None)
    
    for sensor in sensors:
        print("-------SENSOR------", sensor)
        fakeSensorDict = dict.fromkeys([sensor], None)
        bestMethodRMSE = np.inf
        bestMethodEval = methods[0]
        bestMethodParameters = []
        
        for method in methods:
            print("METHOD", method)
            parameters = []
            indexNA = dfWithMissing.loc[dfWithMissing[sensor].isna(), :].index
            
            '''Conditions for parametrics methods'''
            if(method == 'Interpolation'):
                paramHyperopt = paramHyperOptInterpolation
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                interpolationMethod = bestParamConverted['method']
                parameters = [interpolationMethod]
            
            elif(method == 'Moving average'):
                paramHyperopt = paramHyperOptMovingAverage
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                center = bestParamConverted['center']
                window = bestParamConverted['window']
                #print("AASDUADSAUDDAUD", space_eval(paramHyperOptMovingAverage, bestParam))
                minPeriods = bestParam['min_periods']
                #parameters = [window, 1, center]
                parameters = [window, minPeriods, center]
                
            elif(method == 'Flexible moving average'):
                paramHyperopt = paramHyperOptFlexMovingAverage
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method) #<--------- AQUI CHAMO A FUNCAO COM DE CIMA COM O DOMINIO DE PARAMETROS, o treino e o número de iterações
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                window = bestParamConverted['window']
                #print("AASDUADSAUDDAUD", space_eval(paramHyperOptMovingAverage, bestParam))
                #minPeriods = bestParam['min_periods']
                parameters = [window]
                
            elif(method == 'Random forests' and len(sensors)>1):
                #print("ENTRIE..............................")
                paramHyperopt = paramHyperOptRandomForests
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method)
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                maxIter = bestParamConverted['max_iter']
                nEstimators = bestParamConverted['n_estimators']
                minSamplesSplit = bestParamConverted['min_samples_split']
                minSamplesLeaf = bestParamConverted['min_samples_leaf']
                parameters = [int(maxIter), int(nEstimators), int(minSamplesSplit), int(minSamplesLeaf), 0]
                
            elif(method == 'Expectation maximization' and len(sensors)>1):
                paramHyperopt = paramHyperOptEM
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method)
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                loops = bestParamConverted['loops']
                parameters = [int(loops)]
                
            elif(method == 'Knn' and len(sensors)>1):
                paramHyperopt = paramHyperOptKnn
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method)
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                n_neighbors = bestParamConverted['n_neighbors']
                weights = bestParamConverted['weights']
                parameters = [int(n_neighbors), weights]
                
            elif(method == 'Mice' and len(sensors)>1):
                paramHyperopt = paramHyperOptMice
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method)
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                m = bestParamConverted['m']
                defaultMethod = bestParamConverted['defaultMethod']
                maxit = bestParamConverted['maxit']
                norm = bestParamConverted['norm']
                parameters = [int(m), defaultMethod, int(maxit), norm]
                
            elif(method == 'Amelia' and len(sensors)>1):
                paramHyperopt = paramHyperOptAmelia
                bestParam = hyperopt(paramHyperopt, df.copy(), dfWithMissing.copy(), indexNA, MAX_EVALS, sensor, method)
                print("BEST PARAM", bestParam)
                bestParamConverted = space_eval(paramHyperopt, bestParam)
                print("BEST PARAMS CONVERTED", bestParamConverted)
                m = bestParamConverted['m']
                autopri = bestParamConverted['autopri']
                maxresample = bestParamConverted['max.resample']
                norm = bestParamConverted['norm']
                parameters = [int(m), int(autopri), int(maxresample), norm]
                
            fakeSensorDict[sensor] = [method, parameters]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfImputed = imputation_methods.impute_missing_values(dfWithMissing.copy(), fakeSensorDict, 'None', randomState)

            mse, rmse, mae, smape = evaluation.evaluate_singular(df.copy(), dfImputed, dfWithMissing, indexNA)
            #print("RMSE", rmse)
            if(rmse < bestMethodRMSE):
                bestMethodRMSE = rmse
                bestMethodEval = method
                bestMethodParameters = parameters
                
            sensorsDict[sensor] = [bestMethodEval, bestMethodParameters]
            
    return sensorsDict
    


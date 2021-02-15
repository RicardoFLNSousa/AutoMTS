import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from pyts.approximation import SymbolicAggregateApproximation
from scipy.stats import norm
import warnings


# Atypical values Detection Methods



# Outliers Detection Methods

# detect outleirs according to the selected outlier detection method on the Processing options panel on GUI
def detect_outliers(df, sensorDict, outlierDetectionMethodMode, randomSeed):
    
    detectedOutliers = None
    detectedOutliersIndexes = None
    reconstructedDataFinal = df.copy()
    reconstructedDataFinalNan = df.copy()
    outlierDetectionMethodMode = outlierDetectionMethodMode.lower()
    
    for sensor in sensorDict.keys():
        outlierDetectionMethod = sensorDict[sensor][0]
        parameters = sensorDict[sensor][1]
        #print("PARAMETERS", parameters)
        
        if(outlierDetectionMethodMode == 'default'):
            reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = standard_deviation(df.copy(), [], sensor)
        
        else:
            if(outlierDetectionMethod == 'Standard deviation'):
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = standard_deviation(df.copy(), [], sensor)
            elif(outlierDetectionMethod == 'Inter quartile range'):
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = inter_quartile_range(df.copy(), [], sensor)
            elif(outlierDetectionMethod == 'Isolation forests'):
                parameters.append(randomSeed)
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = isolation_forest(df.copy(), parameters, sensor)
                parameters.pop()
            elif(outlierDetectionMethod == 'Local outlier factor'):
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = local_outlier_factor(df.copy(), parameters, sensor)
            elif(outlierDetectionMethod == 'Dbscan'):
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = dbscan(df.copy(), parameters, sensor)
            elif(outlierDetectionMethod == 'K-means'):
                parameters.append(randomSeed)
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = k_means(df.copy(), parameters, sensor)
                parameters.pop()
            elif(outlierDetectionMethod == 'Sax'):
                reconstructedDataFinal[sensor], reconstructedDataFinalNan[sensor] = sax(df.copy(), parameters, sensor)
          

    return reconstructedDataFinal, reconstructedDataFinalNan


# detect outliers using the standard deviation method
def standard_deviation(df, parameters, sensor):    
    # set upper and lower limit to 3 standard deviation
    randomDataStd = np.std(df[sensor])
    randomDataMean = np.mean(df[sensor])
    anomalyCutOff = randomDataStd * 3
    
    lowerLimit = randomDataMean - anomalyCutOff 
    upperLimit = randomDataMean + anomalyCutOff
    # list of indexes of the upper limit outliers
    upperLimitOutliers = df[df[sensor] > upperLimit].index.tolist()
    # list of indexes of the lower limit outliers
    lowerLimitOutliers = df[df[sensor] < lowerLimit].index.tolist()
    
    # concatnate the two lists of outliers indexes
    detectedOutliersIndexes = upperLimitOutliers + lowerLimitOutliers
    # dataframe with detected outliers
    detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
    # list of indexes of the dataframe
    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]
    #return df

def inter_quartile_range(df, parameters, sensor):
    q1 = df[sensor].quantile(0.25)
    q3 = df[sensor].quantile(0.75)
    
    iqr = q3 - q1
    
    lowerLimit = q1 - 1.5 * iqr
    upperLimit = q3 + 1.5 * iqr
    
    # list of indexes of the upper limit outliers
    upperLimitOutliers = df[df[sensor] > upperLimit].index.tolist()
    # list of indexes of the lower limit outliers
    lowerLimitOutliers = df[df[sensor] < lowerLimit].index.tolist()
    
    # concatnate the two lists of outliers indexes
    detectedOutliersIndexes = upperLimitOutliers + lowerLimitOutliers
    # dataframe with detected outliers
    detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
    # list of indexes of the dataframe
    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]

def isolation_forest(df, parameters, sensor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampleSize = df[df.columns[0]].size
        origDf = df.copy()
            
        # transform the data serie into a numpy array then reshape it to fit on the method
        serie = df[sensor].to_numpy().reshape(-1,1)
        
        #print("SERIE", serie)
        
        #returns a list for each observations: -1 is outlier 1 is inlier
        classifier = IsolationForest(n_estimators=parameters[0], max_features=parameters[1], random_state = parameters[-1]).fit_predict(serie) 
        #detectedOutliersIntegerIndexes = np.where(classifier == -1)
        
        origDf['integerIndex'] = classifier
        detectedOutliersIndexes = origDf[origDf['integerIndex'] == -1].index.tolist()
        
        # dataframe with detected outliers
        detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
        # list of indexes of the dataframe
        listOfIndexes = df.index.tolist()
        # list of indexes of all non outliers
        nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
        #turn all outliers values into NaN
        dfWithOutliersNan = df.copy()
        dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
        # turn all non outliers values into NaN
        df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]


def local_outlier_factor(df, parameters, sensor):
    #sampleSize = df[df.columns[0]].size
    origDf = df.copy()
        
    # transform the data serie into a numpy array then reshape it to fit on the method
    serie = df[sensor].to_numpy().reshape(-1,1)
    #returns a list for each observations: -1 is outlier 1 is inlier
    classifier = LocalOutlierFactor(n_neighbors=parameters[0], metric=parameters[1]).fit_predict(serie)
    
    origDf['integerIndex'] = classifier
    detectedOutliersIndexes = origDf[origDf['integerIndex'] == -1].index.tolist()
    
    # dataframe with detected outliers
    detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
    # list of indexes of the dataframe
    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]

def dbscan(df, parameters, sensor):
    sampleSize = df[df.columns[0]].size
    origDf = df.copy()

    # transform the data serie into a numpy array then reshape it to fit on the method
    serie = df[sensor].to_numpy().reshape(-1,1)
    cluster = DBSCAN(eps=parameters[0], min_samples=parameters[1], metric=parameters[2]).fit_predict(serie)
    
    origDf['integerIndex'] = cluster
    detectedOutliersIndexes = origDf[origDf['integerIndex'] == -1].index.tolist()
    
    # dataframe with detected outliers
    detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
    # list of indexes of the dataframe
    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]

def k_means(df, parameters, sensor):
    #sampleSize = df[df.columns[0]].size
    origDf = df.copy()

    # transform the data serie into a numpy array then reshape it to fit on the method
    serie = df[sensor].to_numpy().reshape(-1,1)
    
    cluster = KMeans(n_clusters = 2, random_state = parameters[-1]).fit_predict(serie)
    
    pd.set_option('display.max_rows', df.shape[0]+1)
    
#    print(df[df[sensor] < 2.5].index.tolist())
#    
#    print(cluster)
    
    origDf['integerIndex'] = cluster
    
#    print(origDf)
    detectedOutliersIndexes = origDf[origDf['integerIndex'] == -1].index.tolist()
    
    # dataframe with detected outliers
    detectedOutliers = df.loc[detectedOutliersIndexes, sensor]
    # list of indexes of the dataframe
    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
        
    return df[sensor], dfWithOutliersNan[sensor]

def sax(df, parameters, sensor):
    
    origDf = df.copy()
    serie = origDf[sensor].to_numpy().reshape(1,-1)

    n_bins = 2
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy=parameters[0])
    
    X_sax = sax.fit_transform(serie)[0]
    
    detectedOutliersIndexes = df.loc[X_sax != 'a', sensor].index.tolist()

    listOfIndexes = df.index.tolist()
    # list of indexes of all non outliers
    nonOutliersIndex = list(set(listOfIndexes) - set(detectedOutliersIndexes))
    #turn all outliers values into NaN
    dfWithOutliersNan = df.copy()
    dfWithOutliersNan.loc[detectedOutliersIndexes, sensor] = np.NaN
    # turn all non outliers values into NaN
    df.loc[nonOutliersIndex, sensor] = np.NaN
    
    return df[sensor], dfWithOutliersNan[sensor]
    

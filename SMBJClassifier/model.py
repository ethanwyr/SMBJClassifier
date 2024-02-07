""" Machine Learning (ML) models
This module contains all functions required for the Machine Learning models
Including conversion of conductance traces into 1D/2D histograms, 
turning ML model hyper parameters, and application of XGBoost/CNN+XGBoost 

"""
import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal, optimize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential

def generate_1d_histogram(data, RR, d, inputName, para, 
                          TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum):
    """ Generate 1D histograms by converting conductance traces for the given Dataset
    
    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array (No use in this function)
        1D array with experimental parameters "Ramp Rate" for each Dataset
    d: list of integer
        Specify the indexes of Datasets needed to generate histograms
    inputName: string 
        Name of data files that contain the preprocessed conductance traces 
    para: string
        The parameter that specifies one Dataset, e.g. with R square values, could use 'R95' 
    TEST_FRACTION: float
        The portion of testing data occupied in the entire Dataset
    condEdges: numpy array 
        1D array with conductance edges that used to create 2D histogram
    distEdges: numpy array (No use in this function)
        1D array with distance edges that used to create 2D histogram
    trailNum: integer 
        Number of total histograms, this number is the sum of training data plus testing data
    sampleNum: integer 
        Number of sampled traces to create one histogram
    
    Returns
    -------
    data_train_hist: numpy array
        2D array with generated 1D histograms for training data 
        Each row [i, :] is one sampled conductance histogram, the values are normalized to the sum of 1
    data_test_hist: numpy array
        2D array with generated 1D histograms for testing data 
        Each row [i, :] is one sampled conductance histogram, the values are normalized to the sum of 1

    """
    # load conductance traces of several Datasets
    cond_train = []
    cond_test = []
    for i in d:
        # load conductance traces of one Dataset
        # split the data into training and testing traces 
        matData = loadmat('./Data/' + data[i] + inputName + '_' + para + '.mat')
        Data_LPF = matData['Data_LPF'][0]
        ctr, cte = train_test_split(Data_LPF, test_size=TEST_FRACTION)
        cond_train.extend(ctr)
        cond_test.extend(cte)
    
    # create a random number generator
    rng = np.random.default_rng()

    # create 1D histograms for training data using training traces
    len_a = int(len(d) * trailNum * (1-TEST_FRACTION))
    data_train_hist = np.zeros([len_a, len(condEdges)-1])
    for i in range(len_a): 
        # randomly pick `sampleNum` traces
        idx = rng.integers(len(cond_train), size=sampleNum)
        G = []
        for j in range(sampleNum): 
            cond = cond_train[idx[j]]
            G = np.append(G, cond)
        P, _ = np.histogram(G, bins=condEdges)
        P = P / np.sum(P)
        data_train_hist[i, :] = P
    
    # create 1D histograms for testing data using testing traces
    len_b = int(len(d) * trailNum * TEST_FRACTION)
    data_test_hist = np.zeros([len_b, len(condEdges)-1])
    for i in range(len_b):
        # randomly pick `sampleNum` traces
        idx = rng.integers(len(cond_test), size=sampleNum)
        G = []
        for j in range(sampleNum):
            cond = cond_test[idx[j]]
            G = np.append(G, cond)
        P, _ = np.histogram(G, bins=condEdges)
        P = P / sum(P)
        data_test_hist[i, :] = P
    
    return data_train_hist, data_test_hist

def generate_2d_histogram(data, RR, d, inputName, para, 
                          TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum):
    """ Generate 2D histograms by converting conductance traces for the given Dataset
    
    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    d: integer
        Specify the indexes of Datasets needed to generate histograms
    inputName: string 
        Name of data files that contain the preprocessed conductance traces 
    para: string
        The parameter that specifies one Dataset, e.g. with R square values, could use 'R95' 
    TEST_FRACTION: float
        The portion of testing data occupied in the entire Dataset
    condEdges: numpy array 
        1D array with conductance edges that used to create 2D histogram
    distEdges: numpy array 
        1D array with distance edges that used to create 2D histogram
    trailNum: integer 
        Number of total histograms, this number is the sum of training data plus testing data
    sampleNum: integer 
        Number of sampled traces to create one histogram
    
    Returns
    -------
    data_train_hist: numpy array
        3D array with generated 2D histograms for training data 
        Each row [i, :, :] is one sampled conductance histogram, the values are normalized to the sum of 1
    data_test_hist: numpy array
        3D array with generated 2D histograms for testing data 
        Each row [i, :, :] is one sampled conductance histogram, the values are normalized to the sum of 1 

    """
    # load conductance traces of several Datasets
    cond_train = []
    cond_test = []
    dist_train = []
    dist_test = []
    for i in d:
        # load conductance traces of one Dataset
        # split the data into training and testing traces 
        matData = loadmat('./Data/' + data[i] + inputName + '_' + para + '.mat')
        Data_LPF = matData['Data_LPF'][0]
        ctr, cte = train_test_split(Data_LPF, test_size=TEST_FRACTION)
        cond_train.extend(ctr)
        cond_test.extend(cte)
        dist_train.extend([np.arange(len(c1[0])) * 1e-4 * RR[i] for c1 in ctr])
        dist_test.extend([np.arange(len(c2[0])) * 1e-4 * RR[i] for c2 in cte])
        
    
    # create a random number generator
    rng = np.random.default_rng()

    # create 2D histograms for training data using training traces
    len_a = int(len(d) * trailNum * (1-TEST_FRACTION))
    data_train_hist = np.zeros([len_a, len(condEdges)-1, len(distEdges)-1])
    for i in range(len_a): 
        # randomly pick `sampleNum` traces
        idx = rng.integers(len(cond_train), size=sampleNum)
        G = []
        D = []
        for j in range(sampleNum): 
            cond = cond_train[idx[j]]
            dist = dist_train[idx[j]]
            G = np.append(G, cond)
            D = np.append(D, dist)
        P, _, _ = np.histogram2d(G, D, bins=[condEdges, distEdges])
        P = P / np.sum(P)
        data_train_hist[i, :, :] = P
    
    # create 2D histograms for testing data using testing traces
    len_b = int(len(d) * trailNum * TEST_FRACTION)
    data_test_hist = np.zeros([len_b, len(condEdges)-1, len(distEdges)-1])
    for i in range(len_b):
        # randomly pick `sampleNum` traces
        idx = rng.integers(len(cond_test), size=sampleNum)
        G = []
        D = []
        for j in range(sampleNum): 
            cond = cond_test[idx[j]]
            dist = dist_test[idx[j]]
            G = np.append(G, cond)
            D = np.append(D, dist)
        P, _, _ = np.histogram2d(G, D, bins=[condEdges, distEdges])
        P = P / np.sum(P)
        data_test_hist[i, :, :] = P
    
    return data_train_hist, data_test_hist

def histData(data, RR, group, groupLabel, averageHist, dimension, sampleNum):
    """ Generate Training Data & Label, Testing Data & Label in format of 1D/2D conductance histograms
    
    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    group: list of integer
        2D array with the indexes of Datasets that are prepared for classification 
    groupLabel: list of integer
        1D array with the labels of Datasets that are prepared for classification
        Several Datasets can have the same label
    averageHist: boolean
        True, if creating averaged conductance histograms
    dimension: integer 
        One integer to determine whether generating 1D or 2D histograms
    sampleNum: integer 
        Number of sampled traces to create one histogram
    
    Returns
    -------
    Train_Data: numpy array 
        2D array with 1D conductance histograms for training
        Or 3D array with 2D conductance histograms for training
    Train_Label: numpy array
        1D array with labels of training data
    Test_Data: numpy array
        2D array with 1D conductance histograms for testing
        Or 3D array with 2D conductance histograms for testing
    Test_Label: numpy array
        1D array with labels of testing data

    """
    ## Define the following parameters for constructing histograms 
    inputName = 'Data_LPF'    # Name of data files that contain the preprocessed conductance traces
    R2_value = 'R95'          # R square value for classification 
    TEST_FRACTION = 0.3       # Portion of testing data occupied in the entire Dataset
    trailNum = 500            # Number of total histograms per Dataset
    
    ## Define the following parameters for bin edges of conductance histograms 
    condHigh = -1.5           # Maximum value for the conductance, 10**-1.5 G0
    condLow = -7.5            # Minimum value for the conductance, 10**-7.5 G0
    condBinSize = 600         # Number of bins to use for the conductance bins
    condEdges = 10**(np.linspace(condLow, condHigh, condBinSize+1))

    ## Define the following parameters for bin edges of distance histograms 
    distHigh = 0.4             # Maximum value for the distance, 0.4 nm
    distLow = 0.0              # Minimum value for the distance, 0.0 nm
    distBinSize = 10           # Number of bins to use for the distance bins
    distEdges = np.linspace(distLow, distHigh, distBinSize+1)

    # Determine histogram generation function based on user input 
    if dimension == 1: 
        fun_hist = generate_1d_histogram
    elif dimension == 2:
        fun_hist = generate_2d_histogram
    else:
        print('Please define the dimension of the histograms to be either `1` or `2`.')
        return None

    # Create training and testing histograms for each group label
    Train_Data = []
    Train_Label = []
    Test_Data = []
    Test_Label = []
    if averageHist:
        # Create averaged (probability distribution of) conductance histograms across several Datasets within the same variant
        for gl in groupLabel:
            data_train_hist, data_test_hist = fun_hist(data, RR, group[gl], inputName, R2_value, TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum)
            Train_Data.extend(data_train_hist)
            Train_Label.extend(np.ones(len(data_train_hist), dtype=np.integer) * groupLabel[gl])
            Test_Data.extend(data_test_hist)
            Test_Label.extend(np.ones(len(data_test_hist), dtype=np.integer) * groupLabel[gl])
    else:
        # Create individual conductance histograms for each Dataset within the same variant
        for gl in groupLabel:
            for d in group[gl]:
                data_train_hist, data_test_hist = fun_hist(data, RR, [d], inputName, R2_value, TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum)
                Train_Data.extend(data_train_hist)
                Train_Label.extend(np.ones(len(data_train_hist), dtype=np.integer) * groupLabel[gl])
                Test_Data.extend(data_test_hist)
                Test_Label.extend(np.ones(len(data_test_hist), dtype=np.integer) * groupLabel[gl])

    return Train_Data, Train_Label, Test_Data, Test_Label

def cnn(input_shape, num_class):
    """ The CNN model used in approach A3 and approach A4

    Parameters
    ----------
    input_shape: numpy array
        1D array with the shape of input data for CNN model
    num_class: integer 
        Number of classes (i.e. labels of Datasets) that are prepared for classification
    
    Returns
    -------
    cnn_model: Keras
        A CNN model prepared for later classification
    
    """
    cnn_model = Sequential([
        Conv2D(16, kernel_size = 3, activation = 'relu', padding = 'same', name = 'conv_1', input_shape = input_shape),
        MaxPooling2D(pool_size = 2, padding = 'same', name = 'pool_1'),
        Dropout(0.5, name = 'dropout_1'),
        Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', name = 'conv_2'),
        MaxPooling2D(pool_size = 2, padding = 'same', name = 'pool_2'),
        BatchNormalization(input_shape = input_shape, name = 'norm_1'),
        Dropout(0.5, name = 'dropout_2'),
        Flatten(name = 'flatten'),
        Dense(64, activation = 'relu',name='dense_1'),
        Dense(128, activation = 'relu',name='dense_2')
    ])
    cnn_model.add(Dense(num_class, activation = 'softmax', name = 'dense_classification'))
    cnn_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

def approach_A1(data, RR, group, groupLabel, sampleNum):
    """ Perform classification approach A1, using 1D histograms and the XGBoost model
    Based on user input to construct 1D conductance histograms
    Then use XGBoost model for the training and testing histograms, resulting in confusion matrices
    
    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    group: list of integer
        2D array with the indexes of Datasets that are prepared for classification 
    groupLabel: list of integer
        1D array with the labels of Datasets that are prepared for classification
        Several Datasets can have the same label
    sampleNum: integer 
        Number of sampled traces to create one histogram
    
    Returns
    -------
    conf_mat: numpy array 
        2D array with confusion matrix containing classification result
        Averaged result of 100 iterations

    """
    # Define the parameters for approach A1
    averageHist = False
    dimension = 1

    # Perform the same classification model for 100 iterations
    print('Start classification ...')
    conf_mat = 0
    for simIndex in range(100):
        # Generate 1D histograms
        Train_Data, Train_Label, Test_Data, Test_Label = histData(data, RR, group, groupLabel, averageHist, dimension, sampleNum)

        # Perform classification with XGBoost
        model = XGBClassifier(max_depth=200, n_estimators=2, verbosity = 0)
        model.fit(Train_Data, Train_Label)
        predictedLabels = model.predict(Test_Data)

        # save classification result as confusino matrix
        conf_mat = conf_mat + confusion_matrix(Test_Label, predictedLabels)
        if (simIndex+1) % 10 == 0:
            print('    Finished iteration ', simIndex+1)
    return conf_mat

def cnn_intermediate(cnn_model, layer_name):
    intermediate_layer_model = Model(inputs=cnn_model.input,
                                     outputs=cnn_model.get_layer(layer_name).output)
    return intermediate_layer_model

def xgboost_model(x_train, y_train, num_class, cnn_model = None, layer_name = None, intermediate_layer_model=None):
    if intermediate_layer_model is None:
        intermediate_layer_model = cnn_intermediate(cnn_model, layer_name)
    xgbmodel = XGBClassifier(n_estimators = 2, max_depth = 200, objective='multi:softprob',
                            num_class=num_class, tree_method='hist')
    xgb_train = intermediate_layer_model.predict(x_train)
    xgbmodel.train(xgb_train, y_train)
    return xgbmodel

def xgboost_result(x_test, y_test, xgboost, intermediate_layer_model):
    xgb_test = intermediate_layer_model.predict(x_test)
    xgb_result = xgboost.predict(xgb_test)
    conf_mat = confusion_matrix(xgb_result,y_test)
    return conf_mat

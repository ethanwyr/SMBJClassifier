""" Machine Learning (ML) models
This module contains all functions required for the Machine Learning models
Including conversion of conductance traces into 1D/2D histograms, 
turning ML model hyper parameters, and application of XGBoost/CNN+XGBoost 

"""
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
import xgboost
from xgboost import XGBClassifier
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ReLU
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint

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
        matData = loadmat('./Data/' + data[int(i)] + inputName + '_' + para + '.mat')
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
        matData = loadmat('./Data/' + data[int(i)] + inputName + '_' + para + '.mat')
        Data_LPF = matData['Data_LPF'][0]
        ctr, cte = train_test_split(Data_LPF, test_size=TEST_FRACTION)
        cond_train.extend(ctr)
        cond_test.extend(cte)
        dist_train.extend([np.arange(len(c1[0])) * 1e-4 * RR[int(i)] for c1 in ctr])
        dist_test.extend([np.arange(len(c2[0])) * 1e-4 * RR[int(i)] for c2 in cte])
        
    
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

def histData(data, RR, group, num_group, averageHist, dimension, sampleNum):
    """ Generate Training Data & Label, Testing Data & Label in format of 1D/2D conductance histograms
    
    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    group: list of integer
        2D array with the indexes of Datasets that are prepared for classification 
    num_group: integer
        Number of groups that are prepared for classification
        Several Datasets (under the same group) can have the same label 
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
    num_group = int(num_group)
    dimension = int(dimension)
    sampleNum = int(sampleNum)

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
        for g in range(num_group):
            data_train_hist, data_test_hist = fun_hist(data, RR, group[g], inputName, R2_value, TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum)
            Train_Data.extend(data_train_hist)
            Train_Label.extend(np.ones(len(data_train_hist), dtype=np.integer) * g)
            Test_Data.extend(data_test_hist)
            Test_Label.extend(np.ones(len(data_test_hist), dtype=np.integer) * g)
    else:
        # Create individual conductance histograms for each Dataset within the same variant
        for g in range(num_group):
            for d in group[g]:
                data_train_hist, data_test_hist = fun_hist(data, RR, [d], inputName, R2_value, TEST_FRACTION, condEdges, distEdges, trailNum, sampleNum)
                Train_Data.extend(data_train_hist)
                Train_Label.extend(np.ones(len(data_train_hist), dtype=np.integer) * g)
                Test_Data.extend(data_test_hist)
                Test_Label.extend(np.ones(len(data_test_hist), dtype=np.integer) * g)

    Train_Data = np.array(Train_Data)
    Train_Label = np.array(Train_Label)
    Test_Data = np.array(Test_Data)
    Test_Label = np.array(Test_Label)
    return Train_Data, Train_Label, Test_Data, Test_Label

def cnn(input_shape, num_group):
    """ The CNN model used in approach A3 and approach A4

    Parameters
    ----------
    input_shape: numpy array
        1D array with the shape of input data for CNN model
    num_group: integer 
        Number of groups that are prepared for classification
        Several Datasets (under the same group) can have the same label
    
    Returns
    -------
    cnn_model: Keras
        A CNN model prepared for later classification
    
    """
    cnn_model = Sequential([
        Conv2D(32, kernel_size = 3, padding = 'same', kernel_regularizer = l2(0.0005), input_shape = input_shape),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size = 2, padding = 'same'),
        Conv2D(64, kernel_size = 3, kernel_regularizer = l2(0.0005), padding = 'same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size = 2, padding = 'same'),
        Flatten(name = 'flatten'),
        Dense(64, name='dense_1'),
        ReLU(),
        Dense(32, name='dense_2'),
        ReLU()
    ])
    cnn_model.add(Dense(num_group, activation = 'softmax', name = 'dense_classification'))
    cnn_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

def cnn_pretrain(num_group, Train_Data, Train_Label, Test_Data, Test_Label):
    """ Extracts an intermediate layer of the CNN model
    To obtain features extracted from the intermediate layer of the CNN model
 
    Parameters
    ----------
    num_group: integer
        Number of groups that are prepared for classification
        Several Datasets (under the same group) can have the same label
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
    
    Returns
    -------
    cnn_interLayer_model: Keras
        A CNN model ending at the layer of 'dense_1'
        The intermediate Keras model was used to extract features from the CNN
    
    """
    model = cnn(input_shape = (np.shape(Train_Data)[1], np.shape(Train_Data)[2], 1), num_group=num_group)
    mcp_save = ModelCheckpoint('./Result/', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(Train_Data, Train_Label, epochs = 40, batch_size = 32, callbacks = [mcp_save], 
                        validation_data = (Test_Data, Test_Label), verbose = 0)
    model = load_model('./Result/', compile = False)
    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_interLayer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)

    return cnn_interLayer_model
        
def runClassifier(approach, data, RR, group, num_group, sampleNum):
    """ Perform classification using one of approach A1, A2, A3, or A4
    Using 1D/2D histograms and the XGBoost/XGBoost+CNN model
    
    Parameters
    ----------
    approach:
        The approach going to be used for classification 
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    group: list of integer
        2D array with the indexes of Datasets that are prepared for classification 
    num_group: integer
        Number of groups that are prepared for classification
        Several Datasets (under the same group) can have the same label
    sampleNum: integer 
        Number of sampled traces to create one histogram
    
    Returns
    -------
    conf_mat: numpy array 
        2D array with confusion matrix containing classification result
        Averaged result of 100 iterations

    """
    # Define the parameters for the approach
    approach = int(approach)
    if approach == 1:
        # Approach A1 is 1D histograms with XGBoost model
        averageHist = False
        dimension = 1
    elif approach == 2: 
        # Approach A2 is 1D averaged histograms with XGBoost model
        averageHist = True
        dimension = 1
    elif approach == 3: 
        # Approach A3 is 2D histograms with XGBoost+CNN model
        averageHist = False
        dimension = 2
    elif approach == 4: 
        # Approach A4 is 2D averaged histograms with XGBoost+CNN model
        averageHist = True
        dimension = 2

    # Perform the same classification model for 100 iterations
    print('Start classification ...')
    conf_mat = 0
    for simIndex in range(100):
        # Generate histograms
        Train_Data, Train_Label, Test_Data, Test_Label = histData(data, RR, group, num_group, averageHist, dimension, sampleNum)
    
        # Perform classification with XGBoost/XGBoost+CNN model
        if dimension == 1: 
            # Directly use XGBoost model 
            xgb_model = XGBClassifier(n_estimators=200, max_depth=2, verbosity = 0)
            xgb_model.fit(Train_Data, Train_Label)
            predictedLabels = xgb_model.predict(Test_Data)
        elif dimension == 2:
            # First use CNN model for pre-training
            cnn_interLayer_model = cnn_pretrain(num_group, Train_Data, Train_Label, Test_Data, Test_Label)
            xgb_Train_Data = cnn_interLayer_model.predict(Train_Data, verbose = 0)
            xgb_Test_Data = cnn_interLayer_model.predict(Test_Data, verbose = 0)
            xgb_model = XGBClassifier(objective='multi:softmax', num_class=num_group, tree_method='hist', max_depth=2,verbosity=0, n_estimators=200)

          # Train the model
            xgb_model.fit(xgb_Train_Data, Train_Label, early_stopping_rounds=10)
            predictedLabels = xgb_model.predict(xgb_Test_Data)

        # save classification result as a confusion matrix
        conf_mat = conf_mat + confusion_matrix(Test_Label, predictedLabels)
        if (simIndex+1) % 1 == 0:
            print('    Finished iteration ', simIndex+1)
    
    conf_mat = conf_mat / np.sum(conf_mat[0,:])
    return conf_mat

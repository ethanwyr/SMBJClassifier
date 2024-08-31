""" Piecewise Linear Approximation (PLA)
This module contains all functions required for the Piecewise Linear 
Appoximation method.
PLA could remove noise content from SMBJ condutance traces more effectively 
and keep all useful content unchanged. The resulting conductance histograms, 
which are constructed purely from plateau segments, provide huge benefits on 
training machine learning models. 

"""
import numpy as np
from scipy.io import loadmat, savemat
import random
import multiprocessing
from joblib import Parallel, delayed

def PLFA_cost(x, y, T_max, cost=0.7):
    """ Optimal Piecewise-Linear Function Approximation (PLFA)
    Calculate the piecewise linear approximation of one SMBJ condutance trace 
    using sum of square error SSE to determine the overall approximation error. 
    The additional cost term is for balance between approximation accuracy and 
    number of segments. 
    
    Parameters
    ----------
    x: numpy array
        1D array with "time" values of one SMBJ condutance trace 
    y: numpy array
        1D array with "condutance" values of one SMBJ condutance trace
    T_max: integer
        The maximum number of segments is allowed for one trace
    cost: float
        The cost term for balance between approximation accuracy and number of 
        segments. The default cost term is 0.7. 

    Returns
    -------
    A: numpy array
        2D array with slope, a, of y = ax + b
        Matrix value A[x_i, x_j] is slope for each line with point (x_i, x_j)
    B: numpy array
        2D array with intercept, b, of y = ax + b
        Matrix value B[x_i, x_j] is intercept for each line with point (x_i, x_j)
    E_sum: numpy array
        2D array with sum of approximation errors
        E_sum[x_j, t] is the sum of error for point (x_0, x_j) with "t+1" segments
    P: numpy array
        2D array with break points
        P[x_j, t] is the break point for point (x_0, x_j) with "t+1" segments

    """
    N = len(x)
    assert len(y) == N, f"input x and y do not have the same length, '{x}', '{y}'"
    
    # y = ax + b
    A = np.zeros([N, N])      # A[x_i, x_j] slope matrix for each line with point (x_i, x_j)
    B = np.zeros([N, N])      # B[x_i, x_j] intercept matrix for each line with point (x_i, x_j)
    X = np.zeros([N, N])      # place holder for x variables
    Y = np.zeros([N, N])      # place holder for y variables
    XY = np.zeros([N, N])     # place holder for x*y variables 
    X_sqr = np.zeros([N, N])  # place holder for x^2 variables 

    E = np.zeros([N, N])              # E[x_i, x_j] error matrix for each line with point (x_i, x_j)
    E_sum = np.zeros([N, T_max]) - 1  # E_sum[x_j, t] sum_error matrix, up to point "x_j" with "t+1" segments 
    P = np.zeros([N, T_max]) - 1      # P[x_j, t] break_point matrix, up to point "x_j" with "t+1" segments 

    # create place holder matries
    for i in range(N):
        X[i, i] = x[i]
        Y[i, i] = y[i]
        XY[i, i] = x[i]*y[i]
        X_sqr[i, i] = x[i]**2
        for j in range(i+1, N):
            X[i, j] = X[i, j-1] + x[j]
            Y[i, j] = Y[i, j-1] + y[j]
            XY[i, j] = XY[i, j-1] + x[j]*y[j]
            X_sqr[i, j] = X_sqr[i, j-1] + x[j]**2

    # create error, slope, intercept matries
    for i in range(N):
        # np.inf use as a relative large number for error
        E[i, i] = np.inf
        for j in range(i+1, N):
            # find a linear approximation for (x_i, x_j)
            # a = NΣ(xy) − Σx Σy / N Σ(x2) − (Σx)^2
            n = j - i + 1
            A[i, j] = (n*XY[i, j] - X[i, j]*Y[i, j]) / (n*X_sqr[i, j] - X[i, j]**2)
            B[i, j] = (Y[i, j] - A[i, j]*X[i, j]) / n
            
            ### find the sum of square error SSE ### 
            if i+1 == j:
                E[i, j] = 0
            else:
                E[i, j] = np.sum((y[i:j+1] - A[i, j]*x[i:j+1] - B[i, j])**2)
            ### could switch to different loss function ###
            
    # create sum_error and break_point matrix
    E_sum[:, 0] = E[0, :]
    P[:, 0] = 0
    for t in range(1, T_max):
        ### normal operation for other points ###
        for j in range(N-1, -1, -1):
            if t >= j:
                E_sum[j, t] = np.inf
            else: 
                E_sum[j, t] = E_sum[j, t-1]
                P[j, t] = P[j, t-1]
                for i in range(t, j+1):
                    temp = E_sum[i, t-1] + E[i, j]
                    if E_sum[j, t] > temp:
                        E_sum[j, t] = temp
                        P[j, t] = i
                        
            ### spical operation for last point ###
            if j == N - 1: 
                # only calculate all segment for the last point
                if t == T_max - 1:
                    return [A, B, E_sum, P]
                # early termination for smaller segment error
                if E_sum[j, t]*(cost + 1) >= E_sum[j, t-1]: 
                    ### the cost term play a role ###
                    return [A, B, E_sum, P]
    
    # return all four matrix
    return [A, B, E_sum, P]

def extract_segment(x, y, A, B, E_sum, P):
    """ Successive function after PLFA
    Extract the essential information for all segments after the piecewise linear 
    approximation calculation of one SMBJ condutance trace. Most input parameters 
    are direactly come from PLFA function. 
    
    Parameters
    ----------
    x: numpy array
        1D array with "time" values of one SMBJ condutance trace 
    y: numpy array
        1D array with "condutance" values of one SMBJ condutance trace
    A: numpy array
        2D array with slope, a, of y = ax + b
        Matrix value A[x_i, x_j] is slope for each line with point (x_i, x_j)
    B: numpy array
        2D array with intercept, b, of y = ax + b
        Matrix value B[x_i, x_j] is intercept for each line with point (x_i, x_j)
    E_sum: numpy array
        2D array with sum of approximation errors
        E_sum[x_j, t] is the sum of error for point (x_0, x_j) with "t+1" segments
    P: numpy array
        2D array with break points
        P[x_j, t] is the break point for point (x_0, x_j) with "t+1" segments 

    Returns
    -------
    T: integer
        The number of segments 
    slope: numpy array
        1D array with the slope of each segment, 'T' number of slope in total
    intercept: numpy array
        1D array with the intercept of each segment, 'T' number of intercept in total
    segP: numpy array
        1D array with the break points of each segment. One for starting point and one 
        for ending point. Since segments are connected with each other, there are 'T+1' 
        number of break points in total
    error_seg: numpy array
        1D array with the approximation error of each segment, 'T' number of approximation
        error in total

    """
    N = len(x)
    assert len(y) == N, f"Input x and y do NOT have the same length!"
    
    # determine the number of segments
    error_seg = E_sum[-1, :]
    T = np.where(error_seg <= 0)[0]
    if T.size == 0:
        T = len(error_seg)
    elif T[0] == 0:
        T = T[0] + 1
    else: 
        T = T[0] 
    
    # create place holders
    error_seg = np.zeros([T])
    slope = np.zeros([T])
    intercept = np.zeros([T])
    segP = np.zeros([T])

    # extract the infomation from matrices calculatino from PLFA
    x_i = N - 1
    for t in range(T):
        x_j = x_i
        x_i = int(P[x_j, T-t-1])
        slope[t] = A[x_i, x_j]
        intercept[t] = B[x_i, x_j]
        segP[t] = x_j
        if t == T-1:
            error_seg[t] = E_sum[x_j, T-t-1] - 0
            assert x_i == 0, f"The segments are not start from the first point!"
        else: 
            error_seg[t] = E_sum[x_j, T-t-1] - E_sum[x_i, T-t-2]
    
    # reverse the data into the correct order
    slope = slope[::-1]
    intercept = intercept[::-1]
    segP = segP[::-1]
    error_seg = error_seg[::-1]

    return [T, slope, intercept, segP, error_seg]

def multTrace(data, T_max, cost=0.7):
    """ A helper function parallel computing 
    Combine PLFA functions and return the essential information after the piecewise 
    linear approximation calculation of one SMBJ condutance trace.
    
    Parameters
    ----------
    data: numpy array
        1D array with "condutance" values of one SMBJ condutance trace
    T_max: integer
        The maximum number of segments is allowed for one trace
    cost: float
        The cost term for balance between approximation accuracy and number of 
        segments. The default cost term is 0.7.

    Returns
    -------
    t_t: integer
        The number of segments 
    s_t: numpy array
        1D array with the slope of each segment, 'T' number of slope in total
    i_t: numpy array
        1D array with the intercept of each segment, 'T' number of intercept in total
    p_t: numpy array
        1D array with the break points of each segment. One for starting point and one 
        for ending point. Since segments are connected with each other, there are 'T+1' 
        number of break points in total
    error_seg: numpy array
        1D array with the approximation error of each segment, 'T' number of approximation
        error in total

    """
    y = np.log10(data)
    y = y.flatten()
    # if the trace is too long directly remove it 
    if len(y) > 1000:
        return 1, -10000, len(y), 1000000, 1000000
    
    # x have the unit of time (s)
    x = np.arange(1, len(y)+1) * 0.0001  
    
    # call PLFA calculations
    [A, B, E_sum, P] = PLFA_cost(x, y, T_max, cost)
    [t_t, s_t, i_t, p_t, error_seg] = extract_segment(x, y, A, B, E_sum, P)

    # return only the essential information
    return t_t, s_t, i_t, p_t, error_seg


def PLA(data, Amp, Vbias, drange=None, highCurrent=10.0, lowCurrent=0.001):
    """ Perform Piecewise Linear Approximation (PLA) method to each current traces
    Read 'Data_A.mat' of each Dataset
    Save the calculated segments data in 'Data_CC_segment.mat' under the directory 
    of each Dataset

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    Amp: numpy array
        1D array with experimental parameters "Current Amplifier" for each Dataset
    Vbias: numpy array
        1D array with experimental parameters "Voltage Bias" for each Dataset
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate low pass filtered traces 
    highCurrent: float 
        The saturation current, any current larger than it needs to perform a high clip 
    lowCurrent: float
        The floor current, any current smaller than it needs to perform a low clip 

    """
    # If user not specified, assign `data` length as loop range
    if drange is None:
        drange = range(len(data))

    G0 = 7.748091729 * 10**(-5) # conductance quantum in unit of S 

    # loop through each Dataset    
    for d in drange:
        if data[d] != {}:
            # load raw current traces of one Dataset
            matData = loadmat('./Data/' + data[d] + 'Data_A.mat')  
            Data_A = matData['Data_A'][0]
            Data_CC = [np.array([])] * len(Data_A)

            # loop through each current traces
            for i in range(len(Data_A)):
                current = Data_A[i][:, 1]

                # Perform low and high clip
                cutoffH = np.where(current > highCurrent)[0]
                if len(cutoffH) > 0:
                    current = current[cutoffH[-1]:]
                idx = np.where(current < lowCurrent)[0]
                current[idx] = lowCurrent/1000; 
                cutoff = len(current)
                if len(idx) > 0:
                    cutoff = idx[-1] 
                    if len(idx) > 3:
                        cutoff = idx[2] + 1

                # Create one clipped trace
                current = np.array([current[:cutoff]])
                Data_CC[i] = (Amp[d] / Vbias[d] / G0) * current.T

            # save clipped traces of one Dataset
            Data_CC = np.array(Data_CC, dtype=object)
            savemat('./Data/' + data[d] + 'Data_CC.mat', {'Data_CC':Data_CC})

            # Parallel Loops for PLA calculation
            T_max = 10
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores, verbose=1)( delayed(multTrace)(data, T_max) for data in Data_CC )
            T, slope, intercept, segP, error_seg = zip(*results)

            # save in the correct numpy array format
            slope = np.array(slope, dtype=object)
            intercept = np.array(intercept, dtype=object)
            segP = np.array(segP, dtype=object)
            error_seg = np.array(error_seg, dtype=object)

            # save the segments data for each trace of one Dataset
            savemat('./Data/' + data[d] + 'Data_CC_segment.mat', {'T':T, 'slope':slope, 'intercept':intercept, 'segP':segP, 'error_seg':error_seg})

def keep_flat(data, RR, cs=4.0, drange=None):
    """ Keep the segments that meet cutoff slope criteria (flat segment)
    After performing Piecewise Linear Approximation (PLA) method to each current traces,
    keep the flat segments and remove the noisy segments
    Read 'Data_CC.mat' and 'Data_CC_segment.mat' of each Dataset
    Save the flat segments data in 'Data_CC_x.mat' under the directory of each Dataset,
    where x is the cutoff slope

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    cd: float
        Specify the cutoff slope required for flat segment
        The default cutoff slope term is 4 
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate low pass filtered traces 

    """
    # If user not specified, assign `data` length as loop range
    if drange is None:
        drange = range(len(data))

    # loop through each Dataset    
    for d in drange:
        if data[d] != {}:
            # load condutance traces and their segments of one Dataset
            matData = loadmat('./Data/' + data[d] + 'Data_CC.mat')  
            Data = matData['Data_CC'][0]
            segData = loadmat('./Data/' + data[d] + 'Data_CC_segment.mat') 
            T = segData['T'][0]
            slope = segData['slope'][0]
            intercept = segData['intercept'][0]
            segP = segData['segP'][0]
            error_seg = segData['error_seg'][0]

            # Covert slope of time to slope of distance
            # x of condutance trace have the unit of distance (nm)
            slope_dis = slope/RR[d]/4

            # loop through each condutance traces
            Data_CC = []
            flat_T = []
            for i in range(len(Data)):
                seg_point = np.append([0], segP[i][0])
                seg_idx = np.argwhere(np.abs(slope_dis[i][0]) < cs).T
                T = 0
                # keep only the flat segments
                if seg_idx.size > 0: 
                    one_trace = np.zeros(np.shape(Data[i]))
                    r2 = 0
                    for idx in seg_idx[0]:
                        r1 = int(seg_point[idx])
                        # do not over count the ending points 
                        if r1 == r2 - 1:
                            r1 = r2
                        r2 = int(seg_point[idx+1]) + 1
                        ## skip any below low limit segment
                        if np.mean(Data[i][r1:r2]) < 10**-7.5:
                            continue
                        one_trace[r1:r2] = Data[i][r1:r2]
                        T = T + 1
                # append one trace that contains at least one flat segment
                if T > 0: 
                    Data_CC.extend(one_trace.T)
                    flat_T.append(T)

            # save in the correct numpy array format
            Data_CC = np.array(Data_CC, dtype=object)

            # save the flat segments data for each trace of one Dataset
            savemat('./Data/' + data[d] + 'Data_CC_' + str(cs) + '.mat', {'Data_CC':Data_CC, 'flat_T':flat_T})

def createCondTrace_PLA(data, Amp, Freq, RR, Vbias, drange=None):
    """ Perform data preprocessing
    Convert current traces into conductance traces by applying high/low clip,
    Piecewise Linear Approximation (PLA). 
    Print progress report along the steps. 

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    Amp: numpy array
        1D array with experimental parameters "Current Amplifier" for each Dataset
    Freq: numpy array 
        1D array with experimental parameters "Sampling Frequency" for each Dataset
    RR: numpy array
        1D array with experimental parameters "Ramp Rate" for each Dataset
    Vbias: numpy array
        1D array with experimental parameters "Voltage Bias" for each Dataset
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate conductance traces 

    """
    print('Generating segments with PLA ...')
    PLA(data, Amp, Vbias)
    print('Generating traces with only flat segments ...')
    keep_flat(data, RR)
    print('Finish.')
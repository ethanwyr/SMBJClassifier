""" Data Preprocessing (DPP)
This module contains all functions required for the preprocessing steps.
The main target is converting current traces into conductance traces by 
applying high/low clip, R square value cutoff, and low pass filter. 

"""
import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal, optimize
import warnings

def readInfo(filename, Expfile):
    """ Read the basic information of all Datasets
    
    Parameters
    ----------
    filename: string
        A .txt file containing the directory of all Datasets
        e.g.
        1 = COVID_Alpha/MisMatch_1/Vbias_100mV_Amp_1nAV_RampRate_5_Freq_10kHz/
        2 = COVID_Alpha/MisMatch_1/Vbias_150mV_Amp_10nAV_RampRate_3_Freq_10kHz/
        3 = COVID_Alpha/MisMatch_1/Vbias_100mV_Amp_10nAV_RampRate_10_Freq_10kHz/
    Expfile: string
        A .mat file containing the experimental parameters of all Datasets

    Returns
    -------
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

    """
    data = [{}] * 100
    i = 0
    with open('./Data/' + filename) as datafile: 
        for line in datafile: 
            if '=' in line: 
                line = line.rstrip('\n')
                # line = line.replace('\\', '/') 
                idx = line.find('=') + 2
                i = i+1
                data[i] = line[idx:]
                
    matData = loadmat('./Data/' + Expfile)  
    Amp = np.append(0, matData['Amp'][0])
    Freq = np.append(0, matData['Freq'][0])
    RR = np.append(0, matData['RR'][0])
    Vbias = np.append(0, matData['Vbias'][0])

    return data, Amp, Freq, RR, Vbias

def R2Filter(data, drange=None):
    """ generate R square values for each Dataset
    Fit each current trace to an exponential decay curve and compute its R square value 
    Read 'Data_A.mat' of each Dataset
    Save the R square values in 'Parameter.mat' under the directory of each Dataset

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate R square values  

    """
    # If user not specified, assign `data` length as loop range
    if drange is None:
        drange = range(len(data))
    warnings.filterwarnings("ignore")

    # An exponential decay curve
    fun = lambda x, a, b : a * np.exp(-b * x)

    # loop through each Dataset
    for d in drange:
        if data[d] != {}:
            # load raw current traces of one Dataset
            matData = loadmat('./Data/' + data[d] + 'Data_A.mat')
            Data_A = matData['Data_A'][0]
            rsquare = np.zeros([len(Data_A)])

            # loop through each current traces
            for i in range(len(Data_A)):
                time = Data_A[i][1:, 0]
                curr = Data_A[i][1:, 1]
                try:
                    # Try to perform exponential curve fitting 
                    popt, _ = optimize.curve_fit(fun, time, curr, p0=[1, 10000])

                    # Compute the R square value
                    residuals = curr - fun(time, *popt)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((curr - np.mean(curr))**2)
                    rsquare[i] = 1 - (ss_res / ss_tot)
                except RuntimeError:
                    # When cannot find a good fit, assign R square value to zero
                    rsquare[i] = 0                

            # save R square values of one Dataset
            savemat('./Data/' + data[d] + 'Parameter.mat', {'rsquare':rsquare}) 
  
def LPF_10kHz():
    """ Create a Low Pass Filter (LPF) targeting the sampling frequency of 10kHz

    Returns
    -------
    lpf: numpy array
        1D array containing the coefficients of the optimal LPF for the sampling frequency of 10kHz
    
    """
    Fs  = 10000             # Sample rate
    Fpb = 2600              # End of pass band
    Fsb = 3900              # Start of stop band
    Apb = 0.1               # Max Pass band ripple in dB
    Asb = 70                # Min stop band attenuation in dB
    N   = 24                # Order of the filter (=number of taps-1)

    # Remez weight calculation
    err_pb = (1 - 10**(-Apb/20))/2
    err_sb = 10**(-Asb/20)

    w_pb = 1/err_pb
    w_sb = 1/err_sb
        
    # Calculate that FIR coefficients
    lpf = signal.remez(
        N+1,            # Desired number of taps
        [0., Fpb/Fs, Fsb/Fs, .5], # Filter inflection points
        [1,0],          # Desired gain for each of the bands: 1 in the pass band, 0 in the stop band
        [w_pb, w_sb]    # weights used to get the right ripple and attenuation
        )      
             
    return lpf

def LPF_30kHz():
    """ Create a Low Pass Filter (LPF) targeting the sampling frequency of 30kHz

    Returns
    -------
    lpf: numpy array
        1D array containing the coefficients of the optimal LPF for the sampling frequency of 30kHz
    
    """
    Fs  = 30000             # Sample rate
    Fpb = 8000              # End of pass band
    Fsb = 11000             # Start of stop band
    Apb = 0.1               # Max Pass band ripple in dB
    Asb = 70                # Min stop band attenuation in dB
    N   = 31                # Order of the filter (=number of taps-1)

    # Remez weight calculation
    err_pb = (1 - 10**(-Apb/20))/2
    err_sb = 10**(-Asb/20)

    w_pb = 1/err_pb
    w_sb = 1/err_sb
        
    # Calculate that FIR coefficients
    lpf = signal.remez(
        N+1,            # Desired number of taps
        [0., Fpb/Fs, Fsb/Fs, .5], # Filter inflection points
        [1,0],          # Desired gain for each of the bands: 1 in the pass band, 0 in the stop band
        [w_pb, w_sb]    # weights used to get the right ripple and attenuation
        )      
             
    return lpf

def LPF(data, drange=None, highCurrent=10.0, lowCurrent=0.001):
    """ Create a Low Pass Filter (LPF) and apply it to each current traces
    Read 'Data_A.mat' of each Dataset
    Save the filtered data in 'Data_A_LPF.mat' under the directory of each Dataset

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
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

    # loop through each Dataset    
    for d in drange:
        if data[d] != {}:
            # load raw current traces of one Dataset
            matData = loadmat('./Data/' + data[d] + 'Data_A.mat')
            Data_A = matData['Data_A'][0]
            Data_A_LPF = [np.array([])] * len(Data_A)

            # loop through each current traces
            for i in range(len(Data_A)):
                # Apply LPF to the data
                temp = Data_A[i][:, 1] 
                temp = np.append(np.ones([100])*temp[0], temp)
                current = signal.lfilter(LPF_10kHz(), 1, temp)
                current = current[100:]

                # Perform low and high clip 
                cutoffH = np.where(current > highCurrent)[0]
                if len(cutoffH) > 0:
                    current = current[cutoffH[-1]:]
                idx = np.where(current < lowCurrent)[0]
                cutoff = len(current)
                if len(idx) > 0:
                    cutoff = idx[-1] 
                    if len(idx) > 3:
                        cutoff = idx[2] + 1

                # Create one low pass filtered trace
                current = np.array([current[:cutoff]])
                time = np.array([Data_A[i][:len(current[0]), 0]])
                Data_A_LPF[i] = np.append(time.T, current.T, axis=1)

            # save low pass filtered traces of one Dataset
            Data_A_LPF = np.array(Data_A_LPF, dtype=object)
            savemat('./Data/' + data[d] + 'Data_A_LPF.mat', {'Data_A_LPF':Data_A_LPF}) 

def R2_LPF(data, Amp, Vbias, drange=None, rs_range=range(95, 96)):
    """ Convert current traces into conductance traces 
    by applying an R square cutoff and low pass filter 
    Read 'Data_A_LPF.mat' of each Dataset
    Save the conductance traces in 'Data_LPF_Rxx.mat' under the directory of each Dataset

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    Amp: numpy array
        1D array with experimental parameters "Current Amplifier" for each Dataset
    Vbias: numpy array
        1D array with experimental parameters "Voltage Bias" for each Dataset
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate conductance traces 
    rs_range:  
        Specify the cutoffs of R square values needed to perform on each Dataset
        By default, it uses R square = 0.95, resulting  'Data_LPF_R95.mat'

    """
    # If user not specified, assign `data` length as loop range
    if drange is None:
        drange = range(len(data))

    G0 = 7.748091729 * 10**(-5) # conductance quantum in unit of S 

    # loop through each Dataset
    for d in drange:
        if data[d] != {}:
            # load low pass filtered traces and R square values of one Dataset
            matData = loadmat('./Data/' + data[d] + 'Data_A_LPF.mat')
            Data_A_LPF = matData['Data_A_LPF'][0]
            matData = loadmat('./Data/' + data[d] + 'Parameter.mat')
            rsquare = matData['rsquare'][0]
            Data_LPF = [np.array([])] * len(Data_A_LPF)
            
            # loop through each cutoff R square value in rs_range 
            for rs in rs_range:
                # loop through each LPF traces
                k = 0
                for i in range(len(Data_A_LPF)):
                    if rsquare[i] < rs/100:
                        # If an R square value is smaller than the cutoff
                        # convert current trace to conductance trace
                        current = Data_A_LPF[i][:, 1]
                        Data_LPF[k] = (Amp[d] / Vbias[d] / G0) * current
                        k = k + 1

                # save R square & LPF conductance traces of one Dataset
                Data_LPF = np.array(Data_LPF[:k], dtype=object)
                savemat('./Data/' + data[d] + 'Data_LPF_R' + str(rs) + '.mat', {'Data_LPF':Data_LPF}) 

def createCondTrace(data, Amp, Freq, Vbias, drange=None):
    """ Perform data preprocessing
    Convert current traces into conductance traces by applying high/low clip,
    R square value cutoff, and low pass filter. 
    Print progress report along the steps. 

    Parameters
    ----------
    data: list of string
        Each string in the list contains the directory of one Dataset
    Amp: numpy array
        1D array with experimental parameters "Current Amplifier" for each Dataset
    Freq: numpy array 
        1D array with experimental parameters "Sampling Frequency" for each Dataset
    Vbias: numpy array
        1D array with experimental parameters "Voltage Bias" for each Dataset
    drange: list of integer 
        Specify the indexes for each Dataset needed to generate conductance traces 

    """
    print('Generating R square value ...')
    R2Filter(data, drange)
    print('Generating date with LPF ...')
    LPF(data, drange)
    R2_LPF(data, Amp, Vbias, drange)
    print('Finish.')

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

# Change the working directory to where this file is located 
import os
os.chdir(os.path.dirname(__file__))

# Import SMBJClassifier from upper directory
import sys
sys.path.append('..')
from SMBJClassifier import DPP, model

def main():
    """ Read basic information from text files 
    In this example, we only include COVID Alpha variants
    """
    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = DPP.readInfo(filename, Expfile)


    """ Perform data preprocessing
    Convert current traces into conductance traces by applying high/low clip,
    R square value cutoff, and low pass filter.
    """
    ### This preprocessing step only needs to be performed once ###
    # DPP.createCondTrace(data, Amp, Freq, Vbias)


    """ Perform Machine learning classification 
    Convert conductance traces into 1D/2D histograms
    Then use XGBoost/CNN+XGBoost model for the training and testing histograms
    There are 5 different parameters that could be modified by the user, based on input Datasets
    """
    # 1. Number of groups for classification 
    #    In this example, COVID Alpha has three (groups) different variants
    num_group = 3

    # 2. Name of each group for classification 
    #    In this example, COVID Alpha has Alpha_MM1, Alpha_MM2 Alpha_PM
    group_Label = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM']

    # 3. The indexes of Datasets that are prepared for classification
    #    Write in the format of 2D array, each 1D array contains the Datasets of one group
    #    In this example, COVID Alpha has three groups, and each group has five Datasets
    #    Alpha_MM1 has Datasets E1, E2, E3, E4, E5
    #    Alpha_MM2 has Datasets E6, E7, E8, E9, E10
    #    Alpha_PM has Datasets E11, E12, E13, E14, E15
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

    # 4. The approach going to be used for classification 
    #    Approach A1 is 1D histograms with XGBoost model
    #    Approach A2 is 1D averaged histograms with XGBoost model
    #    Approach A3 is 2D histograms with XGBoost+CNN model
    #    Approach A4 is 2D averaged histograms with XGBoost+CNN model
    #    In the paper, we tested all approaches: [1, 2, 3, 4]
    #    In this example, we can also test one of the approaches: [2]
    approach = [2]

    # 5. Number of sampled traces to create one histogram
    #    An important parameter that could affect the classification result.
    #    In the paper, we tested it with a range of sample sizes: [10, 20, 30, 40, 50]
    #    In this example, we used one sample size: [30]
    sampleNum = [30]

    # Perform classification with each approach and each sample size
    # Save their results in confusion matrices 
    for a in approach:
        for s in sampleNum:
            conf_mat = model.runClassifier(a, data, RR, group, num_group, s)
            savemat('./Result/Alpha_A' + str(a) + '_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'group_Label':group_Label}) 

if __name__ == '__main__':
    main()
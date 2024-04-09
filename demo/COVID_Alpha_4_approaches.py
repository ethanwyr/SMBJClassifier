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
    ### Here are the lines users should modify. ###
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
    ### Here are the lines users should modify. ###
    # 1. Number of groups for classification 
    num_group = 3
    # 2. Name of each group for classification 
    group_Label = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM']
    # 3. The indexes of Datasets that are prepared for classification
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    # 4. The approach going to be used for classification 
    #    In the paper, we tested all approaches: [1, 2, 3, 4]
    approach = [2]
    # 5. Number of sampled traces to create one histogram
    #    In the paper, we tested it with a range of sample sizes: [10, 20, 30, 40, 50]
    sampleNum = [30]

    # Perform classification with each approach and each sample size
    # Save their results in confusion matrices 
    for a in approach:
        for s in sampleNum:
            conf_mat = model.runClassifier(a, data, RR, group, num_group, s)
            savemat('./Result/Alpha_A' + str(a) + '_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'group_Label':group_Label}) 

if __name__ == '__main__':
    main()
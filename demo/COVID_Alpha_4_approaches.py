import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

# change the working directory to where this file is located 
import os
os.chdir(os.path.dirname(__file__))

# import SMBJClassifier from upper directory
import sys
sys.path.append('..')
from SMBJClassifier import DPP, model

def main():
    ## Read basic information from text files 
    # In this example, we only include COVID Alpha variants
    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = DPP.readInfo(filename, Expfile)

    ## Perform data preprocessing
    # Convert current traces into conductance traces by applying high/low clip,
    # R square value cutoff, and low pass filter. 
    ### This preprocessing step only needs to be performed once ###
    #   ------------------------------------------------------   #
    DPP.createCondTrace(data, Amp, Freq, Vbias)
    #   ------------------------------------------------------   #
    

    ## Perform Machine learning classification 
    # Convert conductance traces into 1D/2D histograms
    # Then use XGBoost/CNN+XGBoost model for the training and testing histograms
    
    # First, specify the indexes of three COVID Alpha variants going to classify
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    groupLabel = [0, 1, 2]
    classLabel = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM']

    # An important parameter that could affect the classification result.
    # In the paper, we tested it with a range of sample sizes [10, 20, 30, 40, 50]
    sampleNum = [30]    # Number of sampled traces to create one histogram. 

    # Perform classification with four approaches and save their results in confusion matrices 
    for s in sampleNum:
        conf_mat = model.approach_A1(data, RR, group, groupLabel, s)
        savemat('./Result/Alpha_A1_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'classLabel':classLabel}) 

        # conf_mat = model.approach_A2(data, RR, group, groupLabel, s)
        # savemat('./Result/Alpha_A2_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'classLabel':classLabel}) 

        # conf_mat = model.approach_A3(data, RR, group, groupLabel, s)
        # savemat('./Result/Alpha_A3_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'classLabel':classLabel}) 

        # conf_mat = model.approach_A4(data, RR, group, groupLabel, s)
        # savemat('./Result/Alpha_A4_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'classLabel':classLabel}) 

if __name__ == '__main__':
    main()
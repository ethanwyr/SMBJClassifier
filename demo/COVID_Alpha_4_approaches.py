import numpy as np
from matplotlib import pyplot as plt

# change the working directory to where this file is located 
import os
os.chdir(os.path.dirname(__file__))

# import SMBJClassifier from upper directory
import sys
sys.path.append('..')
from SMBJClassifier import DPP

def main():
    ## Read basic information from text files 
    # In this example, we only include COVID Alpha variants
    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = DPP.readInfo(filename, Expfile)

    ## Perform data preprocessing
    # Convert current traces into conductance traces by applying high/low clip,
    # R square value cutoff, and low pass filter. 
    DPP.createCondTrace(data, Amp, Freq, Vbias)
    
    ## Perform Machine learning classification 
    # Convert conductance traces into 1D/2D histograms, then apply XGBoost/CNN+XGBoost



if __name__ == '__main__':
    main()
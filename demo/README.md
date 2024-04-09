# [COVID Alpha Four Approaches](https://github.com/ethanwyr/SMBJClassifier/tree/main/demo/COVID_Alpha_4_approaches.py) 

A simple demo explains the usage of `SMBJClassifier` package. We choose to use COVID Alpha variants as an example and divide the explanation into three parts: 
- SMBJ raw current traces preparation
- Data preprocessing
- Classification 

More information regarding the overall approach, methods and results can be found in our publication:

## SMBJ raw current traces preparation

The SMBJ data are in the form of raw current traces. One SMBJ experiment could produce thousands of current traces and store them in one `Data_A.mat` file. For preparation, we need three types of files:
- Multiple folders each contain one `Data_A.mat` file (per SMBJ experiment). 
- One `.txt` file contains the path to each folder above. 
- One `.mat` file contains the SMBJ experimental parameters of each SMBJ experiment. 

Then we can read all the basic information with the following function: 

    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = DPP.readInfo(filename, Expfile)

-----------------------

In the paragraphs below, we will explain how we prepared these files in more details. In this example, the SMBJ data of COVID Alpha variants are stored in the following folders under [Data/](https://github.com/ethanwyr/SMBJClassifier/tree/main/demo/Data/). 

    COVID_Alpha/
        MisMatch_1/
            Vbias_100mV_Amp_1nAV_RampRate_5_Freq_10kHz/
                Data_A.mat
            Vbias_150mV_Amp_10nAV_RampRate_3_Freq_10kHz/ 
                Data_A.mat
            Vbias_100mV_Amp_10nAV_RampRate_10_Freq_10kHz/
                Data_A.mat
            …
        MisMatch_2/
        PerfectMatch/

We need to create a `.txt` file containing the path to each folder above. One SMBJ experiment is one Dataset. In this example, we have `Data/COVID_Strand_Source.txt`, and its content has the following format: 
    
    1 = COVID_Alpha/MisMatch_1/Vbias_100mV_Amp_1nAV_RampRate_5_Freq_10kHz/
    2 = COVID_Alpha/MisMatch_1/Vbias_150mV_Amp_10nAV_RampRate_3_Freq_10kHz/
    3 = COVID_Alpha/MisMatch_1/Vbias_100mV_Amp_10nAV_RampRate_10_Freq_10kHz/
    ...

We need to create a `.mat` file containing the SMBJ experimental parameters of each SMBJ experiment. In this example, we can obtain all the information from the naming of each folder. We have `COVID_Exp_parameter.mat`, which includes current amplifier `Amp`, sampling frequency `Freq`, ramp rate `RR`, and voltage bias `Vbias`. 

    Amp = [0, 1.e-09, 1.e-08, 1.e-08, ...... ]
    Freq = [0, 10000, 10000, 10000, ...... ]
    RR = [0, 5, 3, 10, ...... ]
    Vbias = [0, 0.1, 0.15, 0.1, ...... ]

## Data preprocessing

We utilized module `DPP`. This module could convert current traces into conductance traces by applying high/low clip, R square value cutoff, and low pass filter. The preprocessing step only needs to be performed once. We can compulish it by running the following function (once): 

    DPP.createCondTrace(data, Amp, Freq, Vbias)

---------------------
In the paragraphs below, we include a brief introduction about each function called by `DPP.createCondTrace()` and itself. The output files are stored inside the directory of each Dataset. 
- `DPP.createCondTrace()` performs data preprocessing. It also prints progress reports along the steps.
- `DPP.R2Filter()` generates R square values for each Dataset. It fits each current trace to an exponential decay curve and computes its R square value. 
    - Load 'Data_A.mat' 
    - Save 'Parameter.mat'
- `DPP.LPF()` creates a Low Pass Filter (LPF) and applies it to each current traces. 
    - Load 'Data_A.mat' 
    - Save 'Data_A_LPF.mat'
- `DPP.R2_LPF()` converts current traces into conductance traces by applying an R square cutoff and low pass filter. 
    - Load 'Data_A_LPF.mat' and 'Parameter.mat'
    - Save 'Data_LPF_R95.mat'
    
## Classification 

Perform Machine learning classification. Convert conductance traces into 1D/2D histograms. Then use XGBoost/CNN+XGBoost model for the training and testing histograms. There are 5 different parameters that could be modified by the user, based on input Datasets.  

    num_group = 3
    group_Label = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM'] 
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    approach = [2]
    sampleNum = [30]

Then we can easily perform classification with each approach and each sample size (defined above), using two for loops and saving their results in confusion matrices individually. 

    for a in approach:
        for s in sampleNum:
            conf_mat = model.runClassifier(a, data, RR, group, num_group, s)
            savemat('./Result/Alpha_A' + str(a) + '_H' + str(s) + '.mat', {'conf_mat':conf_mat, 'group_Label':group_Label}) 

----------------

In the section below we will explain the selection of the five parameters that could be modified by the user, in great detail. 

1. Number of groups for classification. In this example, COVID Alpha has three (groups) different variants.

        num_group = 3

2. Name of each group for classification. In this example, COVID Alpha has Alpha_MM1, Alpha_MM2, and Alpha_PM.
    
        group_Label = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM']

3. The indexes of Datasets that are prepared for classification. Write in the format of 2D array, each 1D array contains the Datasets of one group. In this example, COVID Alpha has three groups, and each group has five Datasets.  
Alpha_MM1 has Datasets: E1, E2, E3, E4, E5   
Alpha_MM2 has Datasets: E6, E7, E8, E9, E10   
Alpha_PM has Datasets: E11, E12, E13, E14, E15   

        ### group = [group_1, group_2, group_3]
        ### group_1 is Alpha_MM1, group_2 is Alpha_MM2, group_3 is Alpha_PM
        group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

4. The approach going to be used for classification.  
Approach A1 is 1D histograms with XGBoost model.  
Approach A2 is 1D averaged histograms with XGBoost model.  
Approach A3 is 2D histograms with XGBoost+CNN model.  
Approach A4 is 2D averaged histograms with XGBoost+CNN model.  
In the paper, we tested all approaches: [1, 2, 3, 4]. In this example, we can also test one of the approaches: [2]
    
        approach = [2]

5. Number of sampled traces to create one histogram. An important parameter that could affect the classification result. In the paper, we tested it with a range of sample sizes: [10, 20, 30, 40, 50]. In this example, we used one sample size: [30]
    
        sampleNum = [30]
    


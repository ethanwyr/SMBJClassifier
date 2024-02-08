# SMBJClassifier demo

## [COVID Alpha 4 approaches](https://github.com/ethanwyr/SMBJClassifier/tree/main/demo/COVID_Alpha_4_approaches.py) 

A simple demo to explain the usage of `SMBJClassifier` package. We choose to use COVID Alpha variants as an example and divide the explanation into three parts: 
- SMBJ raw current traces preparation
- Data preprocessing
- Classification 

More information regarding the overall approach, methods and results can be found in our publication:

### SMBJ raw current traces preparation

Read basic information from text files

### Data preprocessing

Perform data preprocessing. Convert current traces into conductance traces by applying high/low clip, R square value cutoff, and low pass filter. This preprocessing step only needs to be performed once. 
    
### Classification 

Perform Machine learning classification. Convert conductance traces into 1D/2D histograms. Then use XGBoost/CNN+XGBoost model for the training and testing histograms. There are 5 different parameters that could be modified by the user, based on input Datasets.  
1. Number of groups for classification. 
2. Name of each group for classification 
3. The indexes of Datasets that are prepared for classification
4. The approach going to be used for classification 
5. Number of sampled traces to create one histogram
    
    


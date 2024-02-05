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
import warnings
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:44:33 2020

@author: obazgir
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import r_
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from tqdm import tqdm
import time
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, load_model, model_from_json
from keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import BatchNormalization, Activation, Dropout
from keras.optimizers import Adam, RMSprop, SGD


import csv
import scipy as sp
from scipy.stats import pearsonr
import pickle
from Toolbox import NRMSE, Random_Image_Gen, two_d_norm, two_d_eq, Assign_features_to_pixels, MDS_Im_Gen, Bias_Calc, REFINED_Im_Gen

cell_lines = "HCC_2998"
Results_Dic = {}

#%%

# Loading the the drug responses and their IDs (NSC)
DF = pd.read_csv("NCI60_GI50_normalized_April.csv")
FilteredDF = DF.loc[DF.CELL==cell_lines]											# Pulling out the selected cell line responses
FilteredDF = FilteredDF.drop_duplicates(['NSC'])                                # Dropping out the duplicates


Feat_DF = pd.read_csv("normalized_padel_feats_NCI60_672.csv")	# Load the drug descriptors of the drugs applied on the selected cell line 
Cell_Features = Feat_DF[Feat_DF.NSC.isin(FilteredDF.NSC)]
TargetDF = FilteredDF[FilteredDF.NSC.isin(Cell_Features.NSC)]

Y = np.array(TargetDF.NORMLOG50)
# Features
X = Cell_Features.values
X = X[:,2:]
# fix random seed for reproducibility
seed = 10
np.random.seed(seed)
# split training, validation and test sets based on each sample NSC ID
NSC_All = np.array(TargetDF['NSC'],dtype = int)
Train_Ind, Rest_Ind, Y_Train, Y_Rest = train_test_split(NSC_All, Y, test_size= 0.2, random_state=seed)
Validation_Ind, Test_Ind, Y_Validation, Y_Test = train_test_split(Rest_Ind, Y_Rest, test_size= 0.5, random_state=seed)
# Sort the NSCs
Train_Ind = np.sort(Train_Ind)
Validation_Ind = np.sort(Validation_Ind)
Test_Ind = np.sort(Test_Ind)
# Extracting the drug descriptors of each set based on their associated NSCs
X_Train_Raw = Cell_Features[Cell_Features.NSC.isin(Train_Ind)]
X_Validation_Raw = Cell_Features[Cell_Features.NSC.isin(Validation_Ind)]
X_Test_Raw = Cell_Features[Cell_Features.NSC.isin(Test_Ind)]

Y_Train = TargetDF[TargetDF.NSC.isin(Train_Ind)];  Y_Train = np.array(Y_Train['NORMLOG50']) 
Y_Validation = TargetDF[TargetDF.NSC.isin(Validation_Ind)];  Y_Validation = np.array(Y_Validation['NORMLOG50']) 
Y_Test = TargetDF[TargetDF.NSC.isin(Test_Ind)];  Y_Test = np.array(Y_Test['NORMLOG50']) 

X_Dummy = X_Train_Raw.values;     X_Train = X_Dummy[:,2:]
X_Dummy = X_Validation_Raw.values;     X_Validation = X_Dummy[:,2:]
X_Dummy = X_Test_Raw.values;      X_Test = X_Dummy[:,2:]

Y_Val_Save = np.zeros(((len(Y_Validation)),6))
Y_Val_Save[:,0] = Y_Validation
Y_Test_Save = np.zeros(((len(Y_Test)),6))
Y_Test_Save[:,0] = Y_Test
#%% REFINED coordinates
# LE
import math
with open('REFINED_Coordinates_LE.pickle','rb') as file:
    gene_names_LE,coords_LE,map_in_int_LE = pickle.load(file)
# LLE
with open('REFINED_Coordinates_LLE.pickle','rb') as file:
    gene_names_LLE,coords_LLE,map_in_int_LLE = pickle.load(file)
# ISOMAP
with open('REFINED_Coordinates_Isomap.pickle','rb') as file:
    gene_names_ISO,coords_ISO,map_in_int_ISO = pickle.load(file)
# MDS
with open('REFINED_Coordinates_MDS.pickle','rb') as file:
    gene_names_MDS,coords_MDS,map_in_int_MDS = pickle.load(file)
    
#%% importing tensorflow    
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
Results_Data = np.zeros((5,4))
nn = 26  
cnt = 0                                                               	# Image size = sqrt(#features (drug descriptors))		

# Convert data into images using the coordinates generated by REFINED 
X_Train_REFINED = np.zeros((X_Train.shape[0],nn**2,4))
X_Val_REFINED = np.zeros((X_Validation.shape[0],nn**2,4))
X_Test_REFINED = np.zeros((X_Test.shape[0],nn**2,4))

Temp_Train = REFINED_Im_Gen(X_Train,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Train_REFINED[:,:,0] = REFINED_Im_Gen(X_Train,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Val_REFINED[:,:,0] = REFINED_Im_Gen(X_Validation,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Test_REFINED[:,:,0] = REFINED_Im_Gen(X_Test,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)	

X_Train_REFINED[:,:,1] = REFINED_Im_Gen(X_Train,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Val_REFINED[:,:,1] = REFINED_Im_Gen(X_Validation,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Test_REFINED[:,:,1] = REFINED_Im_Gen(X_Test,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)

X_Train_REFINED[:,:,2] = REFINED_Im_Gen(X_Train,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Val_REFINED[:,:,2] = REFINED_Im_Gen(X_Validation,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Test_REFINED[:,:,2] = REFINED_Im_Gen(X_Test,nn, map_in_int_LE, gene_names_LE,coords_LE)

X_Train_REFINED[:,:,3] = REFINED_Im_Gen(X_Train,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Val_REFINED[:,:,3] = REFINED_Im_Gen(X_Validation,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Test_REFINED[:,:,3] = REFINED_Im_Gen(X_Test,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)

sz = X_Train_REFINED.shape
#Width = int(math.sqrt(sz[1]))
#Height = int(math.sqrt(sz[1]))
Width = 26
Height = 26

X_Train_REFINED = X_Train_REFINED.reshape(-1,Width,Height,4,1)
X_Val_REFINED = X_Val_REFINED.reshape(-1,Width,Height,4,1)
X_Test_REFINED = X_Test_REFINED.reshape(-1,Width,Height,4,1)

def CNN_model(Width,Height,params):
    nb_filters = 64
    nb_conv = 7
    model = models.Sequential()
    # Convlolutional layers
    model.add(layers.Conv3D(int(params['Kernels1']), kernel_size = (int(params['kernel_size1']),int(params['kernel_size1']),int(4)),padding='valid',strides=(int(params['strides1']),int(params['strides1']),1),dilation_rate=1,input_shape=(Width, Height,4,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv3D(int(params['Kernels2']), kernel_size = (int(params['kernel_size2']),int(params['kernel_size2']),int(1)),padding='valid',strides=(int(params['strides2']),int(params['strides2']),1),dilation_rate=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
	
    model.add(layers.Conv3D(int(params['Kernels3']), kernel_size = (int(3),int(3),int(1)),padding='valid',strides=(int(1),int(1),1),dilation_rate=1,input_shape=(Width, Height,4,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))



    model.add(layers.Flatten())
    # Dense layers
    # model.add(layers.Dense(units = int(params['units1'])))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('relu'))

    model.add(layers.Dense(units = int(params['units2'])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(1-0.7))

    model.add(layers.Dense(units = int(params['units3'])))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(1-0.7))

    model.add(layers.Dense(1))

    # optimizer
    initial_learning_rate = params['lr']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=int(params['decay_step']),
        decay_rate=params['decay_rate'],
        staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mse',
    metrics=['mse'])


    return model

#%% Evaluate model function
def evaluate_model(Model, X_Train_REFINED, Y_Train, X_Val_REFINED, Y_Validation,X_Test_REFINED,Y_Test ):
    ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
    History = Model.fit(X_Train_REFINED, Y_Train, batch_size= 128, epochs = 250, verbose=0, validation_data=(X_Val_REFINED, Y_Validation), callbacks = [ES])
    y_pred = Model.predict(X_Test_REFINED)
    CNN_NRMSE, CNN_R2 = NRMSE(Y_Test, y_pred)
    print('NRMSE > %.3f' % (CNN_NRMSE))
    return CNN_NRMSE, History

#%% Hyper parameter tuning
# Defining the hyper parameter space
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
## Hyper parameter grid
param_hyperopt = {
    'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.001)),
    'decay_step' : hp.uniform('decay_step', 5000, 500000),
    'decay_rate': hp.uniform('decay_rate', 0.4, 0.95),
    'Kernels1': hp.uniform('Kernels1', 32, 64),
    'Kernels2': hp.uniform('Kernels2', 32, 128),
    'Kernels3': hp.uniform('Kernels3', 32, 256),
    'kernel_size1': hp.quniform('kernel_size1', 3, 7, 5),
    'kernel_size2': hp.quniform('kernel_size2', 3, 7, 5),
    #'kernel_size3': hp.quniform('kernel_size3', 3, 7, 5),
    'strides1' : hp.quniform('strides1', 1,2,2),
    'strides2' : hp.quniform('strides2', 1,2,2),
    #'strides3' : hp.quniform('strides3', 1,2,2),
    #'units1': hp.quniform('units1', 300, 500, 10),
    'units2': hp.quniform('units2', 80, 400, 20),
    'units3': hp.quniform('units3', 10, 100, 20),
    }
param_space = param_hyperopt

#%% RUN
#y_train = y_train.astype(int)
#y_valid = y_valid.astype(int)
#y_test = y_test.astype(int)
start = time.time()
num_eval = 50
def objective_function(params):
    Width = 26
    Height = 26
    clf = CNN_model(Width,Height,params)
    NRMSE_try, history = evaluate_model(clf, X_Train_REFINED, Y_Train, X_Val_REFINED, Y_Validation,X_Test_REFINED,Y_Test )
    return {'loss': NRMSE_try, 'status': STATUS_OK}

trials = Trials()
best_param = fmin(objective_function, 
                  param_space, 
                  algo=tpe.suggest, 
                  max_evals=num_eval, 
                  trials=trials,
                  rstate= np.random.RandomState(1))
loss = [x['result']['loss'] for x in trials.trials]

best_param_values = [x for x in best_param.values()]

# Retrieve Hyperopt scores
hyperopt_scores = [trial['result']['loss'] for trial in trials]
hyperopt_scores = np.maximum.accumulate(hyperopt_scores)
print("Hyper_opt scores:")
print(hyperopt_scores)

#%% Saving the parameters
import pickle
pickle.dump(trials,open("Trials.p","wb"))
trails = pickle.load(open("Trials.p","rb"))
print("Best parameters: ", best_param)

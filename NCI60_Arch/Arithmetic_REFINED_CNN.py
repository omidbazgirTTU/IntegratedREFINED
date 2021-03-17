# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:50:05 2020

@author: obazgir
"""

import csv
import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import cv2
import pickle
from Toolbox import NRMSE, Random_Image_Gen, two_d_norm, two_d_eq, Assign_features_to_pixels, MDS_Im_Gen, Bias_Calc, REFINED_Im_Gen
from sklearn.metrics import mean_absolute_error
##########################################
#                                        #                                              
#                                        #                               
#               Data Cleaning            #   
#                                        #   
##########################################

cell_lines = ["HCC_2998","MDA_MB_435", "SNB_78", "NCI_ADR_RES","DU_145", "786_0", "A498","A549_ATCC","ACHN","BT_549","CAKI_1","DLD_1","DMS_114","DMS_273","CCRF_CEM","COLO_205","EKVX"]
#cell_lines = ["HCC_2998"]
Results_Dic = {}

SAVE_PATH = "/home/obazgir/REFINED/Volumetric_REFINED/DistanceBased/NCI60/ArithMetricMDSMetr20/"
#%%
for SEL_CEL in cell_lines:
	# Loading the the drug responses and their IDs (NSC)
    DF = pd.read_csv("/home/obazgir/REFINED/NCI/NCI60_GI50_normalized_April.csv")
    FilteredDF = DF.loc[DF.CELL==SEL_CEL]											# Pulling out the selected cell line responses
    FilteredDF = FilteredDF.drop_duplicates(['NSC'])                                # Dropping out the duplicates


    Feat_DF = pd.read_csv("/home/obazgir/REFINED/NCI/normalized_padel_feats_NCI60_672.csv")	# Load the drug descriptors of the drugs applied on the selected cell line 
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
    
    Y_Val_Save = np.zeros(((len(Y_Validation)),2))
    Y_Val_Save[:,0] = Y_Validation
    Y_Test_Save = np.zeros(((len(Y_Test)),2))
    Y_Test_Save[:,0] = Y_Test
    #%% REFINED coordinates
    # LE
    import math
    with open('theMapping_Init_Arith_DistBased20_Metric.pickle','rb') as file:
        gene_names_Geo,coords_Geo,map_in_int_Geo = pickle.load(file)
    #%% importing tensorflow    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    Model_Names = ["Geometric"]
    Results_Data = np.zeros((1,5))
    nn = 26  
    cnt = 0                                                               	# Image size = sqrt(#features (drug descriptors))		
    for modell in Model_Names:

    
        X_Train_REFINED = REFINED_Im_Gen(X_Train,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
        X_Val_REFINED = REFINED_Im_Gen(X_Validation,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
        X_Test_REFINED = REFINED_Im_Gen(X_Test,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)	



        #%% Defining the CNN Model
        
        sz = X_Train_REFINED.shape
        Width = int(math.sqrt(sz[1]))
        Height = int(math.sqrt(sz[1]))

        CNN_Train = X_Train_REFINED.reshape(-1,Width,Height,1)
        CNN_Val = X_Val_REFINED.reshape(-1,Width,Height,1)
        CNN_Test = X_Test_REFINED.reshape(-1,Width,Height,1)
    
        def CNN_model(Width,Height,):
            nb_filters = 64
            nb_conv = 5

            model = models.Sequential()
            # Convlolutional layers
            model.add(layers.Conv2D(54, kernel_size = (nb_conv-1, nb_conv-1),padding='valid',strides=2,dilation_rate=1,input_shape=(Width, Height,1)))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Conv2D(49, kernel_size = (nb_conv+2, nb_conv+2),padding='valid',strides=1,dilation_rate=1))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))

            
            model.add(layers.Flatten())

            
            model.add(layers.Dense(598))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(1-0.7))
            
            model.add(layers.Dense(43))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(1-0.7))
            
            model.add(layers.Dense(1))
            

            initial_learning_rate = 0.0009963583271069686
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=408653,
                decay_rate=0.756065049421736,
                staircase=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mse'])

            #opt = tf.keras.optimizers.Adam(lr=0.0001)
            
            #model.compile(loss='mse', optimizer = opt)
            return model
        # Training the CNN Model
        model = CNN_model(Width,Height)
        ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
        CNN_History = model.fit(CNN_Train, Y_Train, batch_size= 128, epochs = 250, verbose=0, validation_data=(CNN_Val, Y_Validation), callbacks = [ES])
        Y_Val_Pred_CNN = model.predict(CNN_Val, batch_size= 128, verbose=0)
        Y_Pred_CNN = model.predict(CNN_Test, batch_size= 128, verbose=0)
        
        Y_Val_Save[:,cnt+1] = Y_Val_Pred_CNN.reshape(-1)
        Y_Test_Save[:,cnt+1] = Y_Pred_CNN.reshape(-1)
        
        #print(model.summary())
        # Plot the Model
#        plt.plot(CNN_History.history['loss'], label='train')
#        plt.plot(CNN_History.history['val_loss'], label='Validation')
#        plt.legend()
#        plt.show()
        
        # Measuring the REFINED-CNN performance (NRMSE, R2, PCC, Bias)
        CNN_NRMSE, CNN_R2 = NRMSE(Y_Test, Y_Pred_CNN)
        MAE = mean_absolute_error(Y_Test,Y_Pred_CNN)
        print(CNN_NRMSE,"NRMSE of "+ modell + SEL_CEL)
        print(CNN_R2,"R2 of " + modell + SEL_CEL)
        Y_Test = np.reshape(Y_Test, (Y_Pred_CNN.shape))
        CNN_ER = Y_Test - Y_Pred_CNN
        CNN_PCC, p_value = pearsonr(Y_Test, Y_Pred_CNN)
        
        print(CNN_PCC,"PCC of " + modell+ SEL_CEL)
        Y_Validation = Y_Validation.reshape(len(Y_Validation),1)
        Y_Test = Y_Test.reshape(len(Y_Test),1)
        Bias = Bias_Calc(Y_Test, Y_Pred_CNN)
        
        Results_Data[0,:] = [CNN_NRMSE,MAE,CNN_PCC,CNN_R2,Bias]
        cnt +=1
    Results = pd.DataFrame(data = Results_Data , columns = ["NRMSE","MAE","PCC","R2","Bias"], index = Model_Names)
    Y_Val_Save_PD = pd.DataFrame(data = Y_Val_Save , columns = ["Y_Val","Arithmetic"])
    Y_Test_Save_PD = pd.DataFrame(data = Y_Test_Save , columns = ["Y_Val","Arithmetic"])
    
    Y_Val_Save_PD.to_csv(SAVE_PATH + SEL_CEL + "VAL.csv")
    Y_Test_Save_PD.to_csv(SAVE_PATH + SEL_CEL + "TEST.csv")
    Results_Dic[SEL_CEL] = Results
    
print(Results_Dic)

with open(SAVE_PATH+'Results_Dic.csv', 'w') as f:[f.write('{0},{1}\n'.format(key, value)) for key, value in Results_Dic.items()]

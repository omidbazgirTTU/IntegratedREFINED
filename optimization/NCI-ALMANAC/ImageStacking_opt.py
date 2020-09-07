# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:41:25 2020

@author: obazgir
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:42:32 2020

@author: obazgir
"""

import numpy as np
import pandas as pd
import os
import sklearn
import glob
import Toolbox
from Toolbox import NRMSE, Random_Image_Gen, two_d_norm, two_d_eq, Assign_features_to_pixels, MDS_Im_Gen, Bias_Calc, REFINED_Im_Gen
from scipy.stats import pearsonr


from sklearn.metrics import mean_absolute_error
from scipy.stats import sem, t
from scipy import mean
from sklearn.model_selection import train_test_split

def TypeConverter(NSC1):
    yy = []
    for num in NSC1:
        yy.append(int(num))
    yy = np.array(yy[0:])
    return yy

def SetFinder(NSC1_Train,Drug1):
    X_append = []
    for nsc1 in NSC1_Train:
        X_Train_D1 = Drug1[Drug1["NSC"] == nsc1]
        X_append.append(X_Train_D1)
    NonExisting = np.nonzero([xx.shape != (1, 674) for xx in X_append])[0]
    X_Train_D1_PD = pd.concat(X_append)
    return X_Train_D1_PD,NonExisting

NCI_ALM_PD = pd.read_csv("ComboDrugGrowth_Nov2017.csv")
Cells = NCI_ALM_PD["CELLNAME"].unique().tolist()

i = 0
Cell_Inf = NCI_ALM_PD[NCI_ALM_PD["CELLNAME"] == Cells[i]]
UN_NSC1 = Cell_Inf["NSC1"].unique(); UN_NSC1 = UN_NSC1[~np.isnan(UN_NSC1)]; UN_NSC1.dtype = int; UN_NSC1 = UN_NSC1[UN_NSC1 !=0]
UN_NSC2 = Cell_Inf["NSC2"].unique(); UN_NSC2 = UN_NSC2[~np.isnan(UN_NSC2)]; UN_NSC2 = np.array(UN_NSC2,np.int32); UN_NSC2 = UN_NSC2[UN_NSC2 !=0]
append_pd = []
for nsc1 in UN_NSC1:
    for nsc2 in UN_NSC2:
        Temp = Cell_Inf[Cell_Inf["NSC1"] == nsc1]
        Temp2 =Temp[Temp["NSC2"] == nsc2]
        PERCENTGROWTH = np.mean(Temp2["PERCENTGROWTH"])
        if np.isnan(PERCENTGROWTH):
            dumb = 0
        else:
            PERCENTGROWTHNOTZ = np.mean(Temp2["PERCENTGROWTHNOTZ"])
            EXPECTEDGROWTH = np.mean(Temp2["EXPECTEDGROWTH"])
            PANEL = str(Temp2["PANEL"].unique().tolist()).strip("[]")
            Data = np.array([nsc1,nsc2,PERCENTGROWTH,PERCENTGROWTHNOTZ,EXPECTEDGROWTH,PANEL]).reshape(1,-1)
            Target_temp = pd.DataFrame(data = Data, columns = ["NSC1","NSC2","PERCENTGROWTH","PERCENTGROWTHNOTZ","EXPECTEDGROWTH","PANEL"])
            append_pd.append(Target_temp)

Target_PD = pd.concat(append_pd)
Target_PD = Target_PD.reset_index()
Target_PD = Target_PD.drop(['index'],axis = 1)
Target_PD = Target_PD.drop(["PERCENTGROWTHNOTZ"], axis = 1)

#%%

idx = Target_PD.isnull()

Feat_DF = pd.read_csv("normalized_padel_feats_NCI60_672.csv")	# Load the drug descriptors of the drugs applied on the selected cell line 
Drug1 = Feat_DF[Feat_DF.NSC.isin(Target_PD["NSC1"])]
Drug2 = Feat_DF[Feat_DF.NSC.isin(Target_PD["NSC2"])]



y = Target_PD["PERCENTGROWTH"].values.tolist()
yy = []
for num in y:
    yy.append(float(num))
yyy = np.array(yy[0:])
Y = (yyy - yyy.min())/(yyy.max() - yyy.min())


# split training, validation and test sets based on each sample NSC ID
seed = 7
Train_Ind, Rest_Ind, Y_Train, Y_Rest = train_test_split(Target_PD.index.values, Target_PD.index.values, test_size= 0.2, random_state=seed)
Validation_Ind, Test_Ind, Y_Validation, Y_Test = train_test_split(Rest_Ind, Y_Rest, test_size= 0.5, random_state=seed)
# Sort the NSCs
Train_Ind = np.sort(Train_Ind).reshape(-1,1)
Validation_Ind = np.sort(Validation_Ind).reshape(-1,1)
Test_Ind = np.sort(Test_Ind).reshape(-1,1)
# Specifying the traget (observation) values
Y_Train = Y[Train_Ind]; Y_Validation = Y[Validation_Ind]; Y_Test = Y[Test_Ind]
# NSC Train
NSC1 = Target_PD["NSC1"].values.tolist()
NSC1 = TypeConverter(NSC1)
NSC1_Train = NSC1[Train_Ind.tolist()]; NSC1_Train = NSC1_Train.reshape(-1)

NSC2 = Target_PD["NSC2"].values.tolist()
NSC2 = TypeConverter(NSC2)
NSC2_Train = NSC2[Train_Ind.tolist()]; NSC2_Train = NSC2_Train.reshape(-1)

# NSC Validation
NSC1 = Target_PD["NSC1"].values.tolist()
NSC1 = TypeConverter(NSC1)
NSC1_Val = NSC1[Validation_Ind.tolist()]; NSC1_Val = NSC1_Val.reshape(-1)

NSC2 = Target_PD["NSC2"].values.tolist()
NSC2 = TypeConverter(NSC2)
NSC2_Val = NSC2[Validation_Ind.tolist()]; NSC2_Val = NSC2_Val.reshape(-1)

# NSC Test
NSC1 = Target_PD["NSC1"].values.tolist()
NSC1 = TypeConverter(NSC1)
NSC1_Test = NSC1[Test_Ind.tolist()]; NSC1_Test = NSC1_Test.reshape(-1)

NSC2 = Target_PD["NSC2"].values.tolist()
NSC2 = TypeConverter(NSC2)
NSC2_Test = NSC2[Test_Ind.tolist()]; NSC2_Test = NSC2_Test.reshape(-1)

X_Train_D1_PD, NonExTrain1 = SetFinder(NSC1_Train,Drug1); X_Train_D1 = X_Train_D1_PD.values[:,2:]
X_Train_D2_PD, NonExTrain2 = SetFinder(NSC2_Train,Drug2); X_Train_D2 = X_Train_D2_PD.values[:,2:]
NonExTrn = np.union1d(NonExTrain1,NonExTrain2)
Y_Train = np.delete(Y_Train,NonExTrn,axis = 0)
NSC1_Train = np.delete(NSC1_Train, NonExTrn,axis = 0)
NSC2_Train = np.delete(NSC2_Train, NonExTrn,axis = 0)

X_Train_D1_PD, NonExTrain1 = SetFinder(NSC1_Train,Drug1); X_Train_D1 = X_Train_D1_PD.values[:,2:]
X_Train_D2_PD, NonExTrain2 = SetFinder(NSC2_Train,Drug2); X_Train_D2 = X_Train_D2_PD.values[:,2:]



X_Val_D1_PD, NonExVal1 = SetFinder(NSC1_Val,Drug1);    X_Val_D1 = X_Val_D1_PD.values[:,2:]
X_Val_D2_PD, NonExVal2 = SetFinder(NSC2_Val,Drug2);    X_Val_D2 = X_Val_D2_PD.values[:,2:]
NonExVal = np.union1d(NonExVal1,NonExVal2)
Y_Validation = np.delete(Y_Validation, NonExVal,axis = 0)
NSC1_Val = np.delete(NSC1_Val, NonExVal,axis = 0)
NSC2_Val = np.delete(NSC2_Val, NonExVal,axis = 0)

X_Val_D1_PD, NonExVal1 = SetFinder(NSC1_Val,Drug1); X_Val_D1 = X_Val_D1_PD.values[:,2:]
X_Val_D2_PD, NonExVal2 = SetFinder(NSC2_Val,Drug2); X_Val_D2 = X_Val_D2_PD.values[:,2:]


X_Test_D1_PD, NonExTst1 = SetFinder(NSC2_Test,Drug1);  X_Test_D1 = X_Test_D1_PD.values[:,2:]
X_Test_D2_PD, NonExTst2 = SetFinder(NSC2_Test,Drug2);  X_Test_D2 = X_Test_D2_PD.values[:,2:]

NonExTst = np.union1d(NonExTst1,NonExTst2)
Y_Test = np.delete(Y_Test,NonExTst, axis = 0)

NSC1_Test = np.delete(NSC1_Test, NonExTst,axis = 0)
NSC2_Test = np.delete(NSC2_Test, NonExTst,axis = 0)

X_Test_D1_PD, NonExTst1 = SetFinder(NSC1_Test,Drug1); X_Test_D1 = X_Test_D1_PD.values[:,2:]
X_Test_D2_PD, NonExTst2 = SetFinder(NSC2_Test,Drug2); X_Test_D2 = X_Test_D2_PD.values[:,2:]


NonExTst = np.union1d(NonExTst1,NonExTst2)
Y_Test = np.delete(Y_Test,NonExTst, axis = 0)

NSC1_Test = np.delete(NSC1_Test, NonExTst,axis = 0)
NSC2_Test = np.delete(NSC2_Test, NonExTst,axis = 0)

X_Test_D1_PD, NonExTst1 = SetFinder(NSC1_Test,Drug1); X_Test_D1 = X_Test_D1_PD.values[:,2:]
X_Test_D2_PD, NonExTst2 = SetFinder(NSC2_Test,Drug2); X_Test_D2 = X_Test_D2_PD.values[:,2:]
#%% REFINED coordinates
# LE
import math
import pickle
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
from tqdm import tqdm
Results_Data = np.zeros((5,4))
nn = 26  
cnt = 0                      # Image size = sqrt(#features (drug descriptors))		


X_Train_D1_REFINED = np.zeros((X_Train_D1.shape[0],nn**2,4))
X_Val_D1_REFINED = np.zeros((X_Val_D1.shape[0],nn**2,4))
X_Test_D1_REFINED = np.zeros((X_Test_D1.shape[0],nn**2,4))

X_Train_D1_REFINED[:,:,0] = REFINED_Im_Gen(X_Train_D1,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Val_D1_REFINED[:,:,0] = REFINED_Im_Gen(X_Val_D1,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Test_D1_REFINED[:,:,0] = REFINED_Im_Gen(X_Test_D1,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)	

X_Train_D1_REFINED[:,:,1] = REFINED_Im_Gen(X_Train_D1,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Val_D1_REFINED[:,:,1] = REFINED_Im_Gen(X_Val_D1,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Test_D1_REFINED[:,:,1] = REFINED_Im_Gen(X_Test_D1,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)

X_Train_D1_REFINED[:,:,2] = REFINED_Im_Gen(X_Train_D1,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Val_D1_REFINED[:,:,2] = REFINED_Im_Gen(X_Val_D1,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Test_D1_REFINED[:,:,2] = REFINED_Im_Gen(X_Test_D1,nn, map_in_int_LE, gene_names_LE,coords_LE)

X_Train_D1_REFINED[:,:,3] = REFINED_Im_Gen(X_Train_D1,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Val_D1_REFINED[:,:,3] = REFINED_Im_Gen(X_Val_D1,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Test_D1_REFINED[:,:,3] = REFINED_Im_Gen(X_Test_D1,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)



X_Train_D2_REFINED = np.zeros((X_Train_D2.shape[0],nn**2,4))
X_Val_D2_REFINED = np.zeros((X_Val_D2.shape[0],nn**2,4))
X_Test_D2_REFINED = np.zeros((X_Test_D2.shape[0],nn**2,4))

X_Train_D2_REFINED[:,:,0] = REFINED_Im_Gen(X_Train_D2,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Val_D2_REFINED[:,:,0] = REFINED_Im_Gen(X_Val_D2,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)
X_Test_D2_REFINED[:,:,0] = REFINED_Im_Gen(X_Test_D2,nn, map_in_int_ISO, gene_names_ISO,coords_ISO)	

X_Train_D2_REFINED[:,:,1] = REFINED_Im_Gen(X_Train_D2,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Val_D2_REFINED[:,:,1] = REFINED_Im_Gen(X_Val_D2,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)
X_Test_D2_REFINED[:,:,1] = REFINED_Im_Gen(X_Test_D2,nn, map_in_int_MDS, gene_names_MDS,coords_MDS)

X_Train_D2_REFINED[:,:,2] = REFINED_Im_Gen(X_Train_D2,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Val_D2_REFINED[:,:,2] = REFINED_Im_Gen(X_Val_D2,nn, map_in_int_LE, gene_names_LE,coords_LE)
X_Test_D2_REFINED[:,:,2] = REFINED_Im_Gen(X_Test_D2,nn, map_in_int_LE, gene_names_LE,coords_LE)

X_Train_D2_REFINED[:,:,3] = REFINED_Im_Gen(X_Train_D2,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Val_D2_REFINED[:,:,3] = REFINED_Im_Gen(X_Val_D2,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)
X_Test_D2_REFINED[:,:,3] = REFINED_Im_Gen(X_Test_D2,nn, map_in_int_LLE, gene_names_LLE,coords_LLE)


sz = X_Train_D1_REFINED.shape
Width = int(math.sqrt(sz[1]))
Height = int(math.sqrt(sz[1]))

    
CNN_Train_D1 = X_Train_D1_REFINED.reshape(-1,Width,Height,4,1)
CNN_Val_D1 = X_Val_D1_REFINED.reshape(-1,Width,Height,4,1)
CNN_Test_D1 = X_Test_D1_REFINED.reshape(-1,Width,Height,4,1)

CNN_Train_D2 = X_Train_D2_REFINED.reshape(-1,Width,Height,4,1)
CNN_Val_D2 = X_Val_D2_REFINED.reshape(-1,Width,Height,4,1)
CNN_Test_D2 = X_Test_D2_REFINED.reshape(-1,Width,Height,4,1)



def CNN_model(Width,Height,params):


    # ARM 1
    input1 = layers.Input(shape = (Width, Height,4,1))
    x1 = layers.Conv3D(int(params['Kernels1']), kernel_size = (int(params['kernel_size1']),int(params['kernel_size1']),4),padding='valid',strides=(int(params['strides1']),int(params['strides1']),1),dilation_rate=1)(input1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv3D(int(params['Kernels2']), kernel_size = (int(params['kernel_size2']),int(params['kernel_size2']),1),padding='valid',strides=(int(params['strides2']),int(params['strides2']),1),dilation_rate=1)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(int(params['Kernels3']), kernel_size = (1,1),padding='valid',strides=1,dilation_rate=1)(x1)
    Out1 = layers.Flatten()(x1)
    
    input2 = layers.Input(shape = (Width, Height,4,1))
    y1 = layers.Conv3D(int(params['Kernels1']), kernel_size = (int(params['kernel_size1']),int(params['kernel_size1']),4),padding='valid',strides=(int(params['strides1']),int(params['strides1']),1),dilation_rate=1)(input2)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation('relu')(y1)
    y1 = layers.Conv3D(int(params['Kernels2']), kernel_size = (int(params['kernel_size2']),int(params['kernel_size2']),1),padding='valid',strides=(int(params['strides2']),int(params['strides2']),1),dilation_rate=1)(y1)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation('relu')(y1)
    y1 = layers.Conv2D(int(params['Kernels3']), kernel_size = (1,1),padding='valid',strides=1,dilation_rate=1)(y1)
    Out2 = layers.Flatten()(y1)
    
    x = layers.concatenate([Out1, Out2])
    
    x = layers.Dense(units = int(params['units1']))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(1- 0.7)(x)

    
    # x = layers.Dense(units = int(params['units2']))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Dropout(1- 0.7)(x)
    
    Out = layers.Dense(1)(x)
    model = tf.keras.Model(inputs = [input1, input2], outputs = [Out])
    
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
def evaluate_model(Model, CNN_Train_D1,CNN_Train_D2, Y_Train, CNN_Val_D1,CNN_Val_D2, Y_Validation,CNN_Test_D1, CNN_Test_D2,Y_Test ):
    ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=60)
    History = Model.fit([CNN_Train_D1,CNN_Train_D2], Y_Train, batch_size= 128, epochs = 250, verbose=0, validation_data=([CNN_Val_D1,CNN_Val_D2], Y_Validation), callbacks = [ES])
    y_pred = Model.predict([CNN_Test_D1, CNN_Test_D2])
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
    'Kernels1': hp.uniform('Kernels1', 32, 128),
    'Kernels2': hp.uniform('Kernels2', 32, 256),
    'Kernels3': hp.uniform('Kernels3', 1, 128),
    'kernel_size1': hp.quniform('kernel_size1', 3, 7, 5),
    'kernel_size2': hp.quniform('kernel_size2', 3, 7, 5),
    #'kernel_size3': hp.quniform('kernel_size3', 3, 7, 5),
    'strides1' : hp.quniform('strides1', 1,2,2),
    'strides2' : hp.quniform('strides2', 1,2,2),
    #'strides3' : hp.quniform('strides3', 1,2,2),
    #'units1': hp.quniform('units1', 300, 500, 10),
    'units1': hp.quniform('units1', 80, 500, 30),
    'units2': hp.quniform('units2', 10, 100, 30),
    }
param_space = param_hyperopt

#%% RUN
#y_train = y_train.astype(int)
#y_valid = y_valid.astype(int)
#y_test = y_test.astype(int)
import time
start = time.time()
num_eval = 200
def objective_function(params):
    Width = 26
    Height = 26
    clf = CNN_model(Width,Height,params)
    NRMSE_try, history = evaluate_model(clf, CNN_Train_D1,CNN_Train_D2, Y_Train, CNN_Val_D1,CNN_Val_D2, Y_Validation,CNN_Test_D1, CNN_Test_D2,Y_Test )
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
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
from sklearn.metrics import mean_absolute_error


SAVE_PATH = "/home/obazgir/REFINED/Volumetric_REFINED/DistanceBased/NCI_Almanac/ArithMetricMDS/"
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

NCI_ALM_PD = pd.read_csv("/home/obazgir/REFINED/Volumetric_REFINED/NCI_Almanac/ComboDrugGrowth_Nov2017.csv")
Cells = NCI_ALM_PD["CELLNAME"].unique().tolist()

Results_Dic = {}
for i in range(20):
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
    #y = Target_PD["PERCENTGROWTH"].values.tolist()
    #yy = []
    #for num in y:
    #    yy.append(float(num))
    #yyy = np.array(yy[0:])
    #Y = (yyy - yyy.min())/(yyy.max() - yyy.min())
    #
    #Font = 24
    #plt.hist(Y, bins='auto')  # arguments are passed to np.histogram
    #plt.title("Distribution of growth percentage of " + Cells[0] + " cell line", fontsize = Font )
    #plt.ylabel("Count", fontsize = Font)
    #plt.xlabel("Normalized growth percentage", fontsize = Font)   
    #plt.yticks(fontsize = Font)
    #plt.xticks(fontsize = Font)     
    #plt.savefig("C:\\Users\\obazgir\\Desktop\\REFINED project\\Volumetric REFINED\\NCI-Almanac\\Distribution"+ Cells[0]+".pdf")
    #%%
    
    idx = Target_PD.isnull()
    
    Feat_DF = pd.read_csv("/home/obazgir/REFINED/NCI/normalized_padel_feats_NCI60_672.csv")	# Load the drug descriptors of the drugs applied on the selected cell line 
    Drug1 = Feat_DF[Feat_DF.NSC.isin(Target_PD["NSC1"])]
    Drug2 = Feat_DF[Feat_DF.NSC.isin(Target_PD["NSC2"])]
    
    #Desc_NSC = Feat_DF["NSC"].tolist()
    #NSC11111 = Target_PD["NSC1"].tolist()
    #
    #cnt = 0
    #for nsc in NSC11111:
    #    for nsc2 in Desc_NSC:
    #        if nsc == nsc2:
    #            cnt +=1
    #
    #Targ1 = Drug1[Drug1.NSC.isin(Feat_DF["NSC"])]
    #Targ2 = Drug2[Drug2.NSC.isin(Feat_DF["NSC"])]
    
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
    
    #%%
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
    
    Y_Val_Save = np.zeros(((len(Y_Validation)),2))
    Y_Val_Save[:,0] = Y_Validation.reshape(-1)
    Y_Test_Save = np.zeros(((len(Y_Test)),2))
    Y_Test_Save[:,0] = Y_Test.reshape(-1)
    #%% REFINED coordinates
    # LE
    import math
    import pickle
    # Arithmetic
    with open('/home/obazgir/REFINED/Volumetric_REFINED/DistanceBased/theMapping_Init_Arithmetic_DistBased5_Metric.pickle','rb') as file:
        gene_names_Geo,coords_Geo,map_in_int_Geo = pickle.load(file)
        
    #%% importing tensorflow    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    Results_Data = np.zeros((1,5))
    nn = 26  
    cnt = 0                      # Image size = sqrt(#features (drug descriptors))		
    
    MODELS = ["LE","LLE","MDS","ISO"]
    MODELS = ["Arithmetic"]
    for mdl in MODELS:
        if mdl == "Arithmetic":
            X_Train_D1_REFINED = REFINED_Im_Gen(X_Train_D1,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
            X_Val_D1_REFINED = REFINED_Im_Gen(X_Val_D1,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
            X_Test_D1_REFINED = REFINED_Im_Gen(X_Test_D1,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
            
            X_Train_D2_REFINED = REFINED_Im_Gen(X_Train_D2,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
            X_Val_D2_REFINED = REFINED_Im_Gen(X_Val_D2,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
            X_Test_D2_REFINED = REFINED_Im_Gen(X_Test_D2,nn, map_in_int_Geo, gene_names_Geo,coords_Geo)
    
    
    
    
        sz = X_Train_D1_REFINED.shape
        Width = int(math.sqrt(sz[1]))
        Height = int(math.sqrt(sz[1]))
        
            
        CNN_Train_D1 = X_Train_D1_REFINED.reshape(-1,Width,Height,1)
        CNN_Val_D1 = X_Val_D1_REFINED.reshape(-1,Width,Height,1)
        CNN_Test_D1 = X_Test_D1_REFINED.reshape(-1,Width,Height,1)
        
        CNN_Train_D2 = X_Train_D2_REFINED.reshape(-1,Width,Height,1)
        CNN_Val_D2 = X_Val_D2_REFINED.reshape(-1,Width,Height,1)
        CNN_Test_D2 = X_Test_D2_REFINED.reshape(-1,Width,Height,1)
        
        
        
        def CNN_model(Width,Height):
            #nb_filters = 64
            nb_conv = 5
        
            # ARM 1
            input1 = layers.Input(shape = (Width, Height,1))
            x1 = layers.Conv2D(32, (nb_conv, nb_conv),padding='valid',strides=2,dilation_rate=1)(input1)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.Activation('relu')(x1)
            x1 = layers.Conv2D(63, (nb_conv, nb_conv),padding='valid',strides=2,dilation_rate=1)(x1)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.Activation('relu')(x1)
            x1 = layers.Conv2D(39, (1,1),padding='valid',strides=1,dilation_rate=1)(x1)
            Out1 = layers.Flatten()(x1)
            
            input2 = layers.Input(shape = (Width, Height,1))
            y1 = layers.Conv2D(32, (nb_conv, nb_conv),padding='valid',strides=2,dilation_rate=1)(input2)
            y1 = layers.BatchNormalization()(y1)
            y1 = layers.Activation('relu')(y1)
            y1 = layers.Conv2D(63, (nb_conv, nb_conv),padding='valid',strides=2,dilation_rate=1)(y1)
            y1 = layers.BatchNormalization()(y1)
            y1 = layers.Activation('relu')(y1)
            y1 = layers.Conv2D(39, (1,1),padding='valid',strides=1,dilation_rate=1)(y1)
            Out2 = layers.Flatten()(y1)
            
            x = layers.concatenate([Out1, Out2])
            
            x = layers.Dense(100)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(1- 0.7)(x)
        
            
#            x = layers.Dense(40)(x)
#            x = layers.BatchNormalization()(x)
#            x = layers.Activation('relu')(x)
#            x = layers.Dropout(1- 0.7)(x)
            
            Out = layers.Dense(1)(x)
            model = tf.keras.Model(inputs = [input1, input2], outputs = [Out])
            
            initial_learning_rate =  0.0006045089357831582
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=  167458,
                decay_rate=   0.825099598607558,
                staircase=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mse'])
            
            return model
            
        
        model = CNN_model(Width,Height)
        ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=60)
        CNN_History = model.fit([CNN_Train_D1,CNN_Train_D2], Y_Train, batch_size= 128, epochs = 250, verbose=0, validation_data=([CNN_Val_D1,CNN_Val_D2], Y_Validation), callbacks = [ES])
        Y_Val_Pred_CNN = model.predict([CNN_Val_D1,CNN_Val_D2], batch_size= 128, verbose=0)
        Y_Pred_CNN = model.predict([CNN_Test_D1, CNN_Test_D2], batch_size= 128, verbose=0)
        
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
        MAE = mean_absolute_error(Y_Test, Y_Pred_CNN)
        print(CNN_NRMSE,"NRMSE of " + Cells[i])
        print(CNN_R2,"R2 of " +  Cells[i])
        print(MAE,"MAE of " + Cells[i])
        Y_Test = np.reshape(Y_Test, (Y_Pred_CNN.shape))
        CNN_ER = Y_Test - Y_Pred_CNN
        CNN_PCC, p_value = pearsonr(Y_Test, Y_Pred_CNN)
        
        print(CNN_PCC,"PCC of " + Cells[i])
        Y_Validation = Y_Validation.reshape(len(Y_Validation),1)
        Y_Test = Y_Test.reshape(len(Y_Test),1)
        Bias = Bias_Calc(Y_Test, Y_Pred_CNN)
    
        Results_Data[cnt,:] = [CNN_NRMSE,MAE,CNN_PCC,CNN_R2,Bias]
        cnt +=1
    Results = pd.DataFrame(data = Results_Data , columns = ["NRMSE","MAE","PCC","R2","Bias"], index = [Cells[i]])
    Y_Val_Save_PD = pd.DataFrame(data = Y_Val_Save , columns = ["Y_Val","Geo"])
    Y_Test_Save_PD = pd.DataFrame(data = Y_Test_Save , columns = ["Y_Test","Geo"])
    # Name correction
    Cell_Name = Cells[i] 
    if Cell_Name.find("/"):
        Cell_Name = Cell_Name.replace("/", "_")
        
    Y_Val_Save_PD.to_csv(SAVE_PATH + Cell_Name + "VAL.csv")
    Y_Test_Save_PD.to_csv(SAVE_PATH + Cell_Name + "TEST.csv")
    Results_Dic[Cells[i]] = Results

print(Results_Dic)

with open(SAVE_PATH+'Results_Dic.csv', 'w') as f:[f.write('{0},{1}\n'.format(key, value)) for key, value in Results_Dic.items()]

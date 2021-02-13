# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:05:34 2020

@author: Sharmita
"""
#GRU
#from tensorflow.keras import backend
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
from tensorflow.keras import backend

import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import GRU
import tensorflow as tf
from datetime import datetime
import os
from sklearn.metrics import r2_score
import scipy.signal as sp
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import KFold
from tensorflow.python.keras import backend as K
import tensorflow



import pickle
import GP_for_GRU

import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer 

'''dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')'''
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_hidden_nodes = Integer(low=1, high=512, name='num_hidden_nodes')
dim_loss = Categorical(categories=['mae', 'mse'],
                             name='loss')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'tanh'],
                             name='activation')
dim_optimizer = Categorical(categories=['adam', 'sgd'],
                             name='optimizer')
#dim_batch_size = Integer(low=1, high=128, name='batch_size') #dim_batch_size

dimensions = [dim_num_input_nodes,
              dim_num_hidden_nodes,
              dim_loss,
              dim_activation, dim_optimizer]

default_parameters = [500,300, 'mae', 'relu','adam']

#butterworth low pass filter
filtord = 4
lowcutoff = 0.06
fs = 200
mass_subject = 56.5
#highcutoff = 0.0525
b, a = sp.butter(filtord, [lowcutoff], 'low')
def minmaxscale(a, amin, amax):
    for feat in range(a.shape[1]):
        a[:,feat]=(a[:,feat]-amin[feat])/(amax[feat] - amin[feat])
        #a[:,feat]=(a[:,feat])/(amax[feat])
    return a





def create_model(num_input_nodes,
                 num_hidden_nodes, loss, activation, optimizer):
    model = Sequential()
    model.add(GRU(num_input_nodes,activation=activation,return_sequences=True,
                      input_shape=(X_temp.shape[1],
                                   X_temp.shape[2])))#return_sequences=True,
    #model_gru.add(GRU(units=50,activation='tanh'))#, return_sequences=True
    model.add(GRU(units=num_hidden_nodes, activation=activation))
    model.add(Dense(units=1))
    
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
    

####GET DATA

datafolder='D:\Python_codes\Data\Healthy_Walk_RL'
test_datafolder='I:\WALK_datasets\Walk_T\WBDS01walkT'




ntrainsets = np.arange(1, 16)

mass_subject = 56.5

normalize = True
gru_score=[]
lstm_score=[]
gru_trn_time=[]
gru_pred_time=[]

lstm_trn_time=[]
lstm_pred_time=[]
       
append_list_train=[]#name of the list where all the needed cols (input cols) from 10 trials are appended and kept
append_list_output=[]#name of the list where all the needed cols (that serve as output cols) from 10 trials are appended and kept
#append_list_modes = []   

append_list_test= []
append_list_test_out =[]    

for fl1 in os.listdir(datafolder):

    if fl1.endswith('needed_cols.txt'):
        motion_data = pd.read_csv(os.path.join(datafolder,fl1), 
                                              dtype=float, delimiter = '\t')
        
        if fl1.endswith('RL_needed_cols.txt') or fl1.endswith('RL_needed_cols_full_trial.txt') :
            output_label = ['ankle_angle_r'] #, ''ankle_moment_r

            train=['hip_flexion_r','tibia_r_Oz']
            
        elif fl1.endswith('LR_needed_cols.txt') or fl1.endswith('LR_needed_cols_full_trial.txt'):
            output_label = ['ankle_angle_l'] #, ''ankle_moment_r

            train=['hip_flexion_l','tibia_l_Oz']
        container = sp.resample(motion_data[train].values, 200)
        #hip_ang = container[:,0].reshape(-1,1)
        shank_ang = container[:,1] .reshape(-1,1) #*-1 # flexion positive
        
        
        inputs= np.array(shank_ang)
        #inputs = np.concatenate((shank_ang[1:, :]), axis=1) #, , hip_ang_vel
        inputs_filt = sp.filtfilt(b, a, inputs, axis=0)
        append_list_train.append(inputs_filt)
        
        output = sp.resample(motion_data[output_label].values, 200)
        for l in output_label:
            if 'moment' in l:
                output=output/mass_subject
        append_list_output.append((output))

#same structure to acquire test data, since I want to have full trials for testing I am using a different folder to acquire test data where fulltrials are stored

for fl in os.listdir(test_datafolder):

    if fl.endswith('needed_cols_full_trial.txt'):
        motion_data_test = pd.read_csv(os.path.join(test_datafolder,fl), 
                                              dtype=float, delimiter = '\t')
        
        if fl.endswith('RL_needed_cols.txt') or fl.endswith('RL_needed_cols_full_trial.txt') :
            output_label = ['ankle_angle_r'] #, ''ankle_moment_r

            train=['hip_flexion_r','tibia_r_Oz']
            
        elif fl.endswith('LR_needed_cols.txt') or fl.endswith('LR_needed_cols_full_trial.txt'):
            output_label = ['ankle_angle_l'] #, ''ankle_moment_r

            train=['hip_flexion_l','tibia_l_Oz']        
        container_test = motion_data_test[train].values#test data not to be resampled for computational correctness and also to replicate real scenario where resampling is not possible
        
        shank_ang_test = container_test[:,1] .reshape(-1,1) #*-1 # flexion positive  
        inputs_test= np.array(shank_ang_test)
        #inputs = np.concatenate((shank_ang[1:, :]), axis=1) #, , hip_ang_vel
        inputs_filt_test = sp.filtfilt(b, a, inputs_test, axis=0)
        append_list_test.append(inputs_filt_test)
        
        output_test = motion_data_test[output_label].values
        for l in output_label:
            if 'moment' in l:
                output_test=output_test/mass_subject
        append_list_test_out.append((output_test))

        
        
"""train_inp= np.concatenate((append_list_train[0:4], append_list_train[8:12]),axis=0)
train_out= np.concatenate((append_list_output[0:4], append_list_output[8:12]),axis=0)
test_inp= np.concatenate((append_list_train[4:8], append_list_train[12:16]),axis=0)
test_out= np.concatenate((append_list_output[4:8], append_list_output[12:16]),axis=0)"""


train_inp= np.array(append_list_train[0:8:2])
train_out= np.array(append_list_output[0:8:2])
# test_inp= np.array(append_list_train[8:16:2])
# test_out= np.array(append_list_output[8:16:2])

test_inp= np.array(append_list_test[8:16:2])#still having only the last indices as test_outs becoz one cycles from the train indices' trials are used in the training.
test_out= np.array(append_list_test_out[8:16:2])


X_in1= np.tile(np.arange(0,200), len(train_inp)).reshape(-1,1)
y_in1 = np.vstack(train_inp).ravel()
σ_noise = 0.05
X_in2 = np.arange(0,200).reshape(-1,1)
μin2, Σin2 = GP_for_GRU.GP_noise(X_in1, y_in1, X_in2, GP_for_GRU.exponentiated_quadratic, σ_noise)
# Compute the standard deviation at the test points to be plotted
σin2 = np.sqrt(np.diag(Σin2))

# Draw some samples of the posterior
sampled_input = np.random.multivariate_normal(mean=μin2, cov=Σin2, 
                                              size=20)[..., np.newaxis]

y_out1 = np.vstack(train_out).ravel()
σ_noise = 0.05
μout2, Σout2 = GP_for_GRU.GP_noise(X_in1, y_out1, X_in2, GP_for_GRU.exponentiated_quadratic, σ_noise)
# Compute the standard deviation at the test points to be plotted
σout2 = np.sqrt(np.diag(Σout2))

# Draw some samples of the posterior
sampled_output = np.random.multivariate_normal(mean=μout2, cov=Σout2, size=20)[...,np.newaxis]

#putting the actual train data and sampled input data together
complete_inp= np.concatenate((train_inp,sampled_input),axis=0)
complete_out= np.concatenate((train_out,sampled_output),axis=0)


train_trials_num=14
lookback = 10
future_steps=10
data_label_append=[]
gru_pred=[]
lstm_pred=[]

X_sets=[]
Y_sets=[] 

for num_trials in range(train_out.shape[0]):#len(append_list_train)
    # X_sets.extend((append_list_train[num_trials]))
    # Y_sets.extend((append_list_output[num_trials]))
    X_sets.append((sampled_input[num_trials, :].reshape(-1,1)))
    Y_sets.append((sampled_output[num_trials, :].reshape(-1,1)))
    
X_sets=np.array(X_sets)
Y_sets=np.array(Y_sets) 


reshaped_train_inputs=[]
reshaped_train_outputs=[]



for n_trials in range(X_sets.shape[0]):
    Y_all = Y_sets[n_trials][lookback+future_steps:,:]
     
    X_all = np.zeros((len(X_sets[n_trials])-lookback-future_steps, lookback, X_sets[n_trials].shape[1]))
    
    for ii in range(len(X_sets[n_trials])-lookback-future_steps):
        X_all[ii,:,:] = X_sets[n_trials][ii:ii+lookback, :]  
        
    reshaped_train_inputs.extend(X_all)
    reshaped_train_outputs.extend(Y_all)
    
reshaped_train_inputs = np.array(reshaped_train_inputs)
reshaped_train_outputs = np.array(reshaped_train_outputs)
X_temp= reshaped_train_inputs#np.array(X_all)
Y_temp=reshaped_train_outputs#np.array(Y_all)


# scale or normalize train data
train_min = sampled_input.reshape(-1, sampled_input.shape[2]).min(0)
train_max = sampled_input.reshape(-1, sampled_input.shape[2]).max(0)
train_min=train_min.T
train_max=train_max.T

# scale or normalize train data
label_min = (sampled_output.reshape(-1, sampled_output.shape[2]).min(0))
label_max = (sampled_output.reshape(-1, sampled_output.shape[2]).max(0))
label_min=label_min.T
label_max=label_max.T

if normalize:
    for i in range(X_temp.shape[0]):
        X_temp[i]=minmaxscale(X_temp[i], train_min, train_max)
    Y_temp=minmaxscale(Y_temp, label_min, label_max)
    
 
 
#training_sample= (int(len(X_sets[train_index]))-lookback-future_steps)*0.9# using X_sets here, since if i use X_all, and index with train_index, the train_index is more than what x_all contains, x_all is lookback-futuresteps less, since the counter in x_all goes till lokback-future steps
#validation_sample = (int(len(X_sets[train_index]))-lookback-future_steps)*0.1

training_sample= int(len(X_temp)*0.9)
validation_sample = int(len(X_temp)*0.1)
val_data = X_temp[training_sample:training_sample+validation_sample:, :, :]
val_label = Y_temp[training_sample:training_sample+validation_sample:, :]
train_data = X_temp[:training_sample, :, :]
train_label = Y_temp[:training_sample, :]


@use_named_args(dimensions=dimensions)
def fitness(num_input_nodes, num_hidden_nodes, loss, activation, optimizer):
    
    model = create_model(num_input_nodes=num_input_nodes,
                         num_hidden_nodes=num_hidden_nodes,
                         loss=loss,
                         activation=activation,
                         optimizer=optimizer
                        )
    
    #named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_temp,
                        y=Y_temp,
                        epochs=30,
                        batch_size=64,
                        validation_split=0.15,
                        shuffle=False)
    
    #return the validation loss for the last epoch.
    loss = blackbox.history['val_loss'][-1]

    # Print the classification accuracy.
    print()
    print("loss: {0:.4}".format(loss))
    print()


    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    #tensorflow.reset_default_graph()
    
    # the optimizer aims for the lowest score, so we return our negative accuracy
    return loss

gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters)

model_gru = create_model(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4])



strt_time_gru_trn= time.perf_counter()   

gru_summary= model_gru.summary()
   
gru_history = model_gru.fit(X_temp, Y_temp, 
                            epochs=30, batch_size=64, 
                            validation_data=(val_data, val_label), 
                            shuffle=False)
end_time_gru_trn= time.perf_counter() 
gru_trn_time.append(end_time_gru_trn-strt_time_gru_trn)
# pickle_file_gru= ('D:\GRU my_results\saved models\gru'+'gru'+str(test_tria1)+'.pickle')
# with open(pickle_file_gru, 'wb') as handle:
#     pickle.dump(model_gru, handle)#gives TypeError: can't pickle _thread._local objects , using model.save 
#model_gru.save('D:\GRU my_results\saved models\gru'+'_gru_moment'+str(test_tria1)+'.h5')

pyplot.figure(1)
pyplot.plot(gru_history.history['loss'], label='GRU train', color='brown')
pyplot.plot(gru_history.history['val_loss'], label='GRU val', color='#008080')#teal color
pyplot.ylabel('Loss $\\tau_{ankle}$',fontsize=14)
pyplot.xlabel('Epochs',fontsize=14)
# if test_tria1==0:
#     pyplot.legend()
pyplot.show()

#Test data preparation, the same way as train data
     

reshaped_test_inputs=[]
reshaped_test_outputs=[]
  
#reshape test data one by one instead of all together
for num_test_trials in range(test_inp.shape[0]):
    Y_test_sets = test_out[num_test_trials][lookback+future_steps:,:]
     
    X_test_sets = np.zeros((len(test_inp[num_test_trials])-lookback-future_steps, lookback, test_inp[num_test_trials].shape[1]))
    
    for ii in range(len(test_inp[num_test_trials])-lookback-future_steps):
        X_test_sets[ii,:,:] = test_inp[num_test_trials][ii:ii+lookback, :]  
        
    reshaped_test_inputs.extend(X_test_sets)
    reshaped_test_outputs.extend(Y_test_sets)
    
reshaped_test_inputs = np.array(reshaped_test_inputs)
reshaped_test_outputs = np.array(reshaped_test_outputs)
     
     
     

nfold=0

X_test_temp=reshaped_test_inputs
Y_test_temp=reshaped_test_outputs

if normalize:
    for i in range(X_test_temp.shape[0]):
        X_test_temp[i]=minmaxscale(X_test_temp[i], train_min, train_max)
    Y_test_temp=minmaxscale(Y_test_temp, label_min, label_max)
    



#for l in range(len(test_inp)):
strt_time_gru_pred= time.perf_counter() 
gru_single_trl_score= r2_score(model_gru.predict(X_test_temp), Y_test_temp)
end_time_gru_pred= time.perf_counter() 
gru_pred_time.append(end_time_gru_pred-strt_time_gru_pred)

gru_score.append(gru_single_trl_score)
gru_pred.append(model_gru.predict(X_test_temp))
data_label_append.append( Y_test_temp)
    
    
    #lstm model
strt_time_lstm_trn= time.perf_counter()   
   
model_lstm = Sequential()
model_lstm.add(LSTM(500,activation='tanh',return_sequences=True,input_shape=(X_temp.shape[1],X_temp.shape[2])))#return_sequences=True#recurrent_activtion is the inner activation, i.e for creating the gates:it is default hard_sigmoid. and the activation is the the one to produce the candidate gate
#model_lstm.add(LSTM(units=50,activation='tanh'))#return_sequences=True
model_lstm.add(LSTM(units=300, activation='tanh'))#return_sequences=True
model_lstm.add(Dense(units=1))

model_lstm.compile(loss='mae', optimizer='adam')
lstm_summary=model_lstm.summary()   
   
lstm_history = model_lstm.fit(X_temp, Y_temp, epochs=30, batch_size=64, 
                        validation_data=(val_data, val_label), shuffle=False)
end_time_lstm_trn= time.perf_counter() 

lstm_trn_time.append(end_time_lstm_trn-strt_time_lstm_trn)
# pickle_file = os.path.join(project_folder, 'saved_models\Subject-specific model')
# pickle_file = os.path.join(pickle_file, amputee_folder + '_important' + str(nspeeds) + '.pickle')
# pickle_file_lstm= ('D:\GRU my_results\saved models\lstm'+'lstm'+str(test_tria1)+'.pickle')
# with open(pickle_file_lstm, 'wb') as handle:
#     pickle.dump(model_lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)
#model_lstm.save('D:\GRU my_results\saved models\lstm'+'_lstm_moment'+str(test_tria1)+'.h5')
   
pyplot.figure(3)
pyplot.plot(lstm_history.history['loss'], label='lstm train', color='brown')
pyplot.plot(lstm_history.history['val_loss'], label='lstm val', color='#008080')
#pyplot.ylabel('Loss $\\theta_{ankle}$',fontsize=14)
pyplot.xlabel('Epochs',fontsize=14)
   
   
# if test_tria1==0:
#     pyplot.legend()
pyplot.show()

# pyplot.figure(4)
# pyplot.title('lstm prediction')

#pyplot.plot(model_lstm.predict(data_test))
#pyplot.plot(data_label)


#for l in range(len(X_test_temp)):

strt_time_lstm_pred= time.perf_counter() 
   
lstm_single_trl_score=r2_score(model_lstm.predict(X_test_temp), Y_test_temp)
end_time_lstm_pred= time.perf_counter()
lstm_pred_time.append(end_time_lstm_pred-strt_time_lstm_pred)

lstm_score.append(lstm_single_trl_score)
lstm_pred.append(model_lstm.predict(X_test_temp))
    
    
    
    
#plotting the required fields

# gru_score=[]
# lstm_score=[]
# gru_trn_time=[]
# gru_pred_time=[]

# lstm_trn_time=[]
# lstm_pred_time=[]    
    #Done
# gru_pred
# lstm_pred
 
#gru pred visualisation against actual    
list_data_labled_append=[]
for u in  data_label_append: #just to plot the results and take mean and std, i resample all of these to 200,also gru
    list_data_labled_append.extend(u)
data_labled_arr=np.array(list_data_labled_append)
 
#conc_label=np.concatenate(resampled_data_labled_append,axis=1)
pyplot.figure(100)
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.ylabel('$\\theta_{ankle}$',fontsize=16)
pyplot.xlabel('Samples',fontsize=14)


pyplot.plot(data_labled_arr,color='r',linewidth=2)
#pyplot.plot(np.mean(conc_label, axis=1),color='r',linewidth=2)
# pyplot.fill_between(np.arange(0, 200),(np.mean(conc_label, axis=1)+np.std(conc_label, axis=1)),(np.mean(conc_label, axis=1) -np.std(conc_label,axis=1)),color='#ffbbbb')
pyplot.show()

list_gru_pred=[]
for uu in  gru_pred: #just to plot the results and take mean and std, i resample all of these to 200,also gru
    list_gru_pred.extend(uu)
    
gru_pred_arr=np.array(list_gru_pred)
pyplot.figure(100)
pyplot.plot(gru_pred_arr,color='blue')
    
    
#lstm prediction against actual starts here******************
#gru pred visualisation against actual    
#use same resampled_data_label as for gru, since this is ground truth, but plot on a different figure for lstmn, ofcourse
pyplot.figure(101)
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
#pyplot.ylabel('$\\theta_{ankle}$',fontsize=14)


pyplot.plot(data_labled_arr,color='r',linewidth=2)
# pyplot.fill_between(np.arange(0, 200),(np.mean(conc_label, axis=1)+np.std(conc_label, axis=1)),(np.mean(conc_label, axis=1) -np.std(conc_label,axis=1)),color='#ffbbbb')
pyplot.show()

list_lstm_pred=[]
for uuu in  lstm_pred: #just to plot the results and take mean and std, i resample all of these to 200,also gru
    list_lstm_pred.extend(uuu)
lstm_pred_arr = np.array(list_lstm_pred)

pyplot.figure(101)
pyplot.plot(lstm_pred_arr,color='green')    
    
    
# bar graph for r2 score and std dev of gru and lstm
#plt.figure(figsize=(w, h), dpi=d)
pyplot.figure(102) 
positions=[1,1.08]
pyplot.xticks(positions,['GRU','LSTM'],fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.ylabel('$R^2 (\\tau_{ankle})$',fontsize=16)
pyplot.xlabel('Samples',fontsize=14)

means_r2=[np.mean(gru_score, axis=0),np.mean(lstm_score, axis=0)]   
std_r2=[np.std(gru_score, axis=0),np.std(lstm_score, axis=0)]
pyplot.bar(positions, means_r2, color=['blue','green'], yerr=std_r2, width=0.04,align='center',capsize=4)
pyplot.xlim(0.95, 1.13)
#pyplot.savefig('D:\GRU my_results\moment\r2_mom.png')


#bar plot train time
pyplot.figure(103) 
positions=[1,1.08]
pyplot.xticks(positions,['GRU','LSTM'],fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.ylabel('Train time [sec]',fontsize=14)
means_trn_time=[np.mean(gru_trn_time, axis=0),np.mean(lstm_trn_time, axis=0)]   
std_trn_time=[np.std(gru_trn_time, axis=0),np.std(lstm_trn_time, axis=0)]
pyplot.bar(positions, means_trn_time, color=['blue','green'], yerr=std_trn_time, width=0.04,align='center',capsize=4)
pyplot.xlim(0.95, 1.13)
#pyplot.savefig('D:\GRU my_results\moment\trn_time_mom.png')

#bar plot pred time

pyplot.figure(104) 
positions=[1,1.08]
pyplot.xticks(positions,['GRU','LSTM'],fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.ylabel('Pred time [sec]',fontsize=14)
means_pred_time=[np.mean(gru_pred_time, axis=0),np.mean(lstm_pred_time, axis=0)]   
std_pred_time=[np.std(gru_pred_time, axis=0),np.std(lstm_pred_time, axis=0)]
pyplot.bar(positions, means_pred_time, color=['blue','green'], yerr=std_pred_time, width=0.04,align='center',capsize=4)
pyplot.xlim(0.95, 1.13)
#pyplot.ylim(0.0, 0.20)
#pyplot.savefig('D:\GRU my_results\moment\pred_time_mom.png')


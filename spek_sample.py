# CUDA_VISIBLE_DEVICES=0 python spek_cnn.py
# works with tensorflow 2.2
#
import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt
import time

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
#from tensorflow.keras import models
#from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers import Reshape
#import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error as mse

# Save Path
dir_path = "/home/jadolphs/spektren/deep_learn/"
log_dir  = "/home/jadolphs/spektren/deep_learn/logs"

print("TF-Version:       ", tf.__version__)
print("Keras-Version:    ", tf.keras.__version__)
#print("Sklearn-Version:  ", sklearn.__version__)
print()

SPECIES = 'AES' #'TEP' #'AES'
TRAINSET = '_coup_0' # '_coup_45'

EPOCHS = 200 #100#50 
LOSS = 'mse'# 'mae' #'mse'
LR = 0.001 #0.001 # 0.001
OPTIMIZER = optimizers.Adam(learning_rate = LR) 
DROPOUT = 0.3
SAMPLE = 10#20#10 #50  # Loop over ... trainings- and prediction cycles

ES_PAT = 10#10 #5 # early-stopping-patience 

LR_START = 0.001 # start-laerning_rate for scheduler
BATCH_SIZE = 1024 #128

t_start = time.time()

#------------------------------------------------------------------------------

names_1 = ['spek_nr', 'site_1', 'site_2', 'site_3', 'site_4', 'site_5', 'site_6', 'site_7', 'site_8']
names_2 = ['spek_nr', 'w_cm', 'OD', 'CD', 'LD']
names_3 = ['w_cm', 'amplitude']

# Import and process data
df1 = pd.read_csv("siteEner_"+SPECIES+TRAINSET+".dat", sep = " ", names = names_1) # train siteEnerg
df2 = pd.read_csv( "spectra_"+SPECIES+TRAINSET+".dat", sep = " ", names = names_2) # train spectra
df3 = pd.read_csv("exp_spek_"+SPECIES+".dat", sep = " ", names = names_3)    # Experimental Spek

df1.drop(['spek_nr'], axis = 1, inplace = True)
df1.drop(['site_3'],  axis = 1, inplace = True) # remove site 3 from search space

y = df1.values              # Site Energies, Target!

X1 = df2["OD"].values  # Spectra
X2 = df2["CD"].values
X3 = df2["LD"].values

n_spek = y.shape[0]  # number of spectra

X1 = np.reshape(X1, (n_spek, 354))
X2 = np.reshape(X2, (n_spek, 354))
X3 = np.reshape(X3, (n_spek, 354))

# 3D-Matrix with (H,B,T) = (n_spek, 177, 3) 
X = np.dstack((X1, X2, X3))

#------------------------------------------------------------------------------
# Experimental-spectrum

X_real_1 = df3["amplitude"].values[:354]
X_real_2 = df3["amplitude"].values[354:708]
X_real_3 = df3["amplitude"].values[708:]

X_real_1 = np.reshape(X_real_1, (1, 354))
X_real_2 = np.reshape(X_real_2, (1, 354))
X_real_3 = np.reshape(X_real_3, (1, 354))

X_real = np.dstack((X_real_1, X_real_2, X_real_3))

#y_real_tep = pd.DataFrame([12445, 12520, 12205, 12335, 12490, 12640, 12450, 12800]).T # TEP
#y_real_aes = pd.DataFrame([12475, 12460, 12225, 12405, 12665, 12545, 12440, 12800]).T # AES

#------------------------------------------------------------------------------

scores = []
pred_list = []
test_err_mean_list = []
test_err_std_list = []

for i in range(SAMPLE): 
    
    # split into train, validation and test set  [70% / 15% / 15%]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size = 0.2)#0.1) # 0.3
    X_val, X_test,  y_val, y_test  = train_test_split(X_tmp, y_tmp, test_size = 0.1)#0.01)  # 0.5)
    
    # Transform site energies from [12000, 13000] to [0, 1] with y=(y-12000)*0.001 
    y_train = (y_train - 12000)*0.001 
    y_val   = (y_val   - 12000)*0.001 
    y_test  = (y_test  - 12000)*0.001 
    
    # Features scaling (spectra) to [0, 1]
    scaler = MaxAbsScaler()
    for jj in range(3):
        X_train[:,:, jj] = scaler.fit_transform(X_train[:,:, jj])
        X_val[:, :, jj] = scaler.transform(X_val[:, :, jj])
        X_test[:,:, jj] = scaler.transform(X_test[:,:, jj])


    
    #------------------------------------------------------------------------------
    # initializer
    # Conv1D and Dense uses GloroUniform as default initializer with seed=None
    # initGloroUni = tf.keras.initializers.GlorotNormal(seed=13)
    # kernel_initializer=initGloroUn, 
    #------------------------------------------------------------------------------
    model = Sequential()
    
    model.add(Conv1D(filters=16, kernel_size=7, activation='relu', input_shape = (354, 3) )) # 5, 7
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu' ))                         # 5
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu' ))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu' ))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu' ))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu' ))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    
    model.add(Dense(units = 768, activation='relu')) #256
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    
    model.add(Dense(units = 256, activation='relu')) # 128
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    
    model.add(Dense(units = 128, activation='relu')) #64
    model.add(BatchNormalization())
    model.add(Dropout(rate=DROPOUT))
    
    #model.add(Dense(units = 8)) # output, no activation function in regression # 8 Sites
    model.add(Dense(units = 7)) # output, no activation function in regression # 7 Sites
    
    model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = ['accuracy']) # 0.003
    
    if (i == 0): 
        model.summary()
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
               
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LR_START*10**(-epoch/(EPOCHS)))   # -3*epoch
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PAT)
    
    history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                        shuffle=True, verbose = 0, #2,  
                        validation_data = (X_val, y_val),   
                        callbacks = [es]) #, [es, tb, lr_scheduler])
    
    score = model.evaluate(X_test, y_test, batch_size = BATCH_SIZE)

    scores.append(score[0])
    print()
    print("Test loss:  ", score[0])
    print()
    
    # Transform site energies from [12000, 13000] to [0, 1] with y=(y-12000)*0.001 
    y_train = y_train*1000 + 12000 
    y_val   = y_val*1000   + 12000 
    y_test  = y_test*1000  + 12000 
    
    #------------------------------------------------------------------------------
    
    y_test_pred = model.predict(X_test)     #, batch_size = 32)
    y_test_pred = y_test_pred*1000 + 12000
    
    y_real_pred = model.predict(X_real)
    y_real_pred = y_real_pred*1000 + 12000

    pred_list.append(y_real_pred)

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    print()
    ##for i in range(5):    
    ##    print("test:      ", np.round(y_test[i, :], 0))
    ##    print("test_pred: ", np.round(y_test_pred[i, :], 0))
    ##    print()
    print('----------------------------------------------------------------------')
    
    test_err_mean = np.round(np.mean(np.abs(y_test_pred-y_test), axis=0))
    test_err_std  = np.round(np.std(np.abs(y_test_pred-y_test), axis=0))
    
    test_err_mean_list.append(test_err_mean)
    test_err_std_list.append(test_err_std)
    
    print()
    print("Mean Test Error "+SPECIES+": ", test_err_mean)
    print("Std Test Error "+SPECIES+":  ", test_err_std)
    print()
    print('----------------------------------------------------------------------')
    print()
    #print("real_tep: ", np.round(np.array(y_real_tep), 0))
    #print("real_aes: ", np.round(np.array(y_real_aes), 0))
    #print()
    print("pred_"+SPECIES+":  ", np.round(y_real_pred[0, :], 0))
    print()

print('====================================================================')
print()
print(scores)
print()
print("min(score):  ", min(scores))
print("index(min):  ", scores.index(min(scores)))
print()
print('====================================================================')
print()
print(np.round(np.array(pred_list)))
print()
print('====================================================================')
print()
print("Average_mean_test_error  ", np.round(np.array(test_err_mean_list).mean(axis=0)))
print("Average_std_test_error   ", np.round(np.array(test_err_std_list).mean(axis=0)))
print("StanDev_all_Samples      ", np.round(np.array(pred_list).std(axis=0)))
print()
print("Average_all_Samples      ", np.round(np.array(pred_list).mean(axis=0)))
print()
print("Training-Set:   ", SPECIES+TRAINSET)
print()
print(LR, LOSS, ES_PAT)
print()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Train- and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Train- and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 0.05)
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Train- and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 0.01)
plt.legend(loc='upper right')
plt.show()


#------------------------------------------------------------------------------ 
t_end = time.time()
t_run = (t_end - t_start) / 60
print(" --- %s minutes --- " % t_run )   
#------------------------------------------------------------------------------


'''
plt.figure()
plt.plot(X_train[0, :, 0],  label='X_train')
plt.plot(X_train[1, :, 0],  label='X_train')
plt.plot(X_train[2, :, 0],  label='X_train')
plt.plot(X_train[3, :, 0],  label='X_train')
plt.plot(X_train[4, :, 0],  label='X_train')
plt.title('Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('No')
#plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(X_real[0, :, 0],  label='X_real')
plt.title('Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('No')
plt.show()

i=0 
plt.figure(i)
'''
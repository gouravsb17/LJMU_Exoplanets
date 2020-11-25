# Importing the required libraries
import os
import cv2
import timeit
import random
import shutil
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import lightkurve as lk
from numpy import save,load
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from keras.layers import Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed
from keras.layers import LSTM

os.chdir('..')

tqdm.pandas(desc="Progress: ")
warnings.filterwarnings('ignore')

SPLIT_PERCENT = 0.75
CROP_FLAG = True
randomize_bool = True
SEED_VALUE = 1

# Model Params
train_size, validation_size, test_size = 5000, 1000, 1000
BATCH_SIZE = 32
PER_TRAIN_SAMPLE = 20
EPOCH = 20
VALIDATION_STEP = 5
THRESHOLD = 0.5
CLASS_WEIGHT = {0:1.0,
                1:1.0}

# Paths
TS_res_path = 'res/KIC_flux_time_series_data/'
model_save_path = 'res/model/'

try:
    os.mkdir(TS_res_path)
except:
    print(len(os.listdir(TS_res_path)))
    print('Folder already exists...reading')
    KIC_array_list = [x.split('_')[-1].split('.')[0].strip(' ') for x in os.listdir(TS_res_path)]

def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)
    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)
    return lc_nan_normalize_flatten

def generate_time_series_fn(kepler_id):
    kepler_id = str(kepler_id).strip(' ')
    if('kepler_ID_'+str(kepler_id)+'.npy' in os.listdir(TS_res_path)):
        return 'Time Series already present'

    # Pulling from the External HDD to the temp resource folder
    try:
        res_path = '/Volumes/PaligraphyS/kepler_data/res/kepler_ID_'+kepler_id+'/'
        shutil.copytree(res_path,'temp_res/kepler_ID_'+kepler_id+'/')
        res_path = 'temp_res/kepler_ID_'+kepler_id+'/'
    except:
        return 'Data for KIC - '+ kepler_id +' not in Ext HDD'

    lc_list_files = []
    for lc_file in os.listdir(res_path):
        if('llc.fits' in lc_file):
            lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile(res_path+lc_file))

    lc_collection = lk.LightCurveFileCollection(lc_list_files)
    stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

    corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)

    # Removing the kepler data brought to the temporary directory
    shutil.rmtree('temp_res/kepler_ID_'+kepler_id)

    save(TS_res_path + 'kepler_ID_' + kepler_id+'.npy', np.array(corrected_lc.to_pandas()[['time', 'flux']]))
    return 'Time series sent'

def multiprocessor_plot_fn(data_df,generate_TCE_fn):

    temp_dir = "res/tempDir"

    KIC_list = []
    for iter, rows in data_df.iterrows():
        KIC_list.append((rows['KIC']))

    cores = multiprocessing.cpu_count()*2

    def worker(sub_trackers_list, start):
        plotted_df = pd.DataFrame(columns=['KIC','Array'])
        k = start
        for sub_tracker in tqdm(sub_trackers_list):
            plotted_df.loc[k] = [sub_tracker, generate_TCE_fn(sub_tracker)]
            k += 1
        plotted_df.to_csv(temp_dir + "/plotted_df_" + str(start) + ".csv", sep=",", index=False)
        return

    slot = int(np.ceil(len(KIC_list) / cores))
    end = 0

    # Timer start
    start_time = timeit.default_timer()

    for i in range(cores):
        start = end
        if (start == slot * (cores - 1)):
            end = len(KIC_list)
        else:
            end = start + slot
        sub_trackers_list = KIC_list[start:end]
        p = multiprocessing.Process(target=worker, args=(sub_trackers_list, start,))
        p.start()
        # Statement to not exit un-till the process is complete for p
        p.join()

    # Timer end
    elapsed = timeit.default_timer() - start_time

    final_KIC_plotted_df = pd.DataFrame(columns=['KIC','Array'])
    for i in os.listdir(temp_dir):
        final_KIC_plotted_df = pd.concat([final_KIC_plotted_df, pd.read_csv(temp_dir + "/" + i)],
                                             ignore_index=True,sort=False)
        os.remove(temp_dir + "/" + i)

    print('Time Elapsed : '+ str(elapsed))

    return (final_KIC_plotted_df)

#### TASK 1: Getting the train and test data
# Reading the data showing which KIC has planets and which not
complete_kepler_df = pd.read_csv('planetary_data/planetary_data_kepler_mission.csv',sep=',',dtype={'kepid':str})
complete_kepler_df = complete_kepler_df[['kepid','nconfp','nkoi','nkoi']]

# Reading the metadata for which the data has been downloaded
metadata_df = pd.read_csv('planetary_data/kepler_download_metadata.csv',sep=',',dtype={'kic':str})
metadata_df = metadata_df.loc[(metadata_df['download']=='Y') & (metadata_df['fits_count']>=14)]

# new_TCE_Array = []
# for kic in metadata_df['kic'].values:
#     if(kic not in KIC_array_list):
#         new_TCE_Array.append(kic)
#
# print(len(new_TCE_Array))
#
# # exit()
#
# data_df = pd.DataFrame(columns=['KIC'])
# data_df['KIC'] = new_TCE_Array
# multiprocessor_plot_fn(data_df,generate_time_series_fn)
#
# exit()

# Merging the data that has been downloaded and the planet info df
complete_kepler_df = pd.merge(complete_kepler_df,metadata_df,left_on='kepid',right_on='kic',how='inner')

# Merging the data on the KIC ID's which have been plotted
plot_df = pd.DataFrame(columns=['KIC'])
plot_df['KIC'] = KIC_array_list

complete_kepler_df = pd.merge(complete_kepler_df,plot_df,left_on='kepid',right_on='KIC',how='inner')

# Separating the data into KIC having planets and not having any confirmed planets
confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']>0.0]
no_confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']==0.0]

#### TASK 2: Preparing the train and test data
# Getting the list of Kepler ID
confirmed_planets_KIC_list = confirmed_planet_df['kepid'].values
no_confirmed_planets_KIC_list = no_confirmed_planet_df['kepid'].values

confirmed_planets_KIC_dict = {}
for kic in confirmed_planets_KIC_list:
  confirmed_planets_KIC_dict[kic] = 1

no_confirmed_planets_KIC_dict = {}
for kic in no_confirmed_planets_KIC_list:
  no_confirmed_planets_KIC_dict[kic] = 0

print('Total KIC with confirmed planets vs non confirmed planets = ',len(confirmed_planets_KIC_list),'/',len(no_confirmed_planets_KIC_list))

KIC_list = np.array(list(confirmed_planets_KIC_list)+list(no_confirmed_planets_KIC_list))
random.seed(SEED_VALUE)
random.shuffle(KIC_list)

kic_train_list = list(KIC_list)[0:train_size]
kic_val_list = list(KIC_list)[train_size:train_size+validation_size]
kic_test_list = list(KIC_list)[train_size+validation_size:train_size+validation_size+test_size]

y_train = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_train_list]
y_val = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_val_list]
y_test = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_test_list]

print(len(kic_train_list),"Train Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_train_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_train_list]))

print(len(kic_val_list),"Validation Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_val_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_val_list]))

print(len(kic_test_list),"Test Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_test_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_test_list]))

def array_fn(kepler_id):
    array_data = load(TS_res_path + 'kepler_ID_' + kepler_id + '.npy')[:,1]
    if (array_data.shape[0] < 67000):
        return np.pad(array_data, (1, 66999 - array_data.shape[0]), 'constant')
    else:
        return array_data[0:66999, :]

def generator(samples, batch_size = 32):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        np.random.shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X)
                array_data = array_fn(batch_sample[0])
                # Read label (y)
                y = batch_sample[1]
                # Add example to arrays
                X_train.append(array_data)
                y_train.append(float(y))

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

            # The generator-y part: yield the next training batch
            yield X_train, y_train

# Import list of train and validation data (image filenames and image labels)
train_samples = np.array(list(map(lambda x,y: [x,y], kic_train_list,y_train)))
validation_samples = np.array(list(map(lambda x,y: [x,y], kic_val_list,y_val)))

# Create generator
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

data_rows,data_cols = 67000,1

### Time Distributed CNN Model
# model = Sequential()
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,data_rows,data_cols)))
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
# model.add(TimeDistributed(Dropout(0.5)))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(10))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

## CNN Model
model = Sequential()
model.add(Conv1D(64, kernel_size=3,activation='relu',input_shape=(data_rows,data_cols)))
model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(10))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights(model_save_path+"model_checkpoint.h5")

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())

# define the checkpoint
filepath = model_save_path+"model_checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit model using generator
model.fit(train_generator,
          steps_per_epoch=len(train_samples)//PER_TRAIN_SAMPLE,
          epochs=EPOCH,
          validation_data=validation_generator,
          validation_steps=VALIDATION_STEP,
          class_weight=CLASS_WEIGHT,
          callbacks=callbacks_list
          )

model_json = model.to_json()
if(CROP_FLAG):
    model_name = 'model-CNN-TS-67000-RUN-1'
else:
    model_name = 'model-CNN-TS-67000-RUN-1'

with open(model_save_path + model_name+".json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_save_path + model_name+".h5")
print("Saved model to disk")

### TEST on un seen data
# load json and create model
json_file = open(model_save_path + model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(model_save_path + model_name+".h5")
print("Loaded model from disk")

# Compile the model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# Predicting on the test set
test_X_samples = np.array(list(map(lambda x: array_fn(x),kic_test_list)))
test_X_samples = test_X_samples.reshape(test_X_samples.shape[0],test_X_samples.shape[1],1)
predit_output = loaded_model.predict(test_X_samples)

print(list(predit_output))
print(len(predit_output))

y_pred = list(map(lambda x: 0+1*(x>=THRESHOLD), predit_output))
y_true = y_test

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

print('Test Results: -')
print('True Negative (No Confirmed Planets) = '+str(tn(y_true,y_pred)),
      'False Positive = '+str(fp(y_true, y_pred)),
      'False Negative = '+str(fn(y_true, y_pred)),
      'True Positive (Confirmed Planets) = '+str(tp(y_true, y_pred)))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

average_precision = average_precision_score(y_test, predit_output)
print(average_precision)

precision, recall, thresholds = precision_recall_curve(y_test, predit_output)

plt.plot(recall, precision,'-')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

# roc curve and auc on an imbalanced dataset
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = predit_output
# keep probabilities for the positive outcome only
# lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('CNN: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CNN')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

print('Complete')


# Recommended actions by Yun
# 1) Try to use Feed forward NN for time series data - Baseline
# 2) RNN/LSTM
# 3) Try to achieve results close enough to existing ones

# #### TASK 1: Getting the train and test data
# # Reading the data showing which KIC has planets and which not
# complete_kepler_df = pd.read_csv(res_path+'planetary_data/planetary_data_kepler_mission.csv',sep=',',dtype={'kepid':str})
# complete_kepler_df = complete_kepler_df[['kepid','nconfp','nkoi','nkoi']]
#
# # Reading the metadata for which the data has been downloaded
# metadata_df = pd.read_csv(res_path+'planetary_data/kepler_download_metadata.csv',sep=',',dtype={'kic':str})
# metadata_df = metadata_df.loc[(metadata_df['download']=='Y') & (metadata_df['fits_count']>=14)]
#
# # Merging the data that has been downloaded and the planet info df
# complete_kepler_df = pd.merge(complete_kepler_df,metadata_df,
#                                left_on='kepid',right_on='kic',how='inner')
#
# # Merging the data on the KIC ID's which have been plotted
# plot_df = pd.DataFrame(columns=['KIC'])
# plot_df['KIC'] = kic_list_downloaded
#
# complete_kepler_df = pd.merge(complete_kepler_df,plot_df,left_on='kepid',
#                               right_on='KIC',how='inner')
#
# # Separating the data into KIC having planets and not having any confirmed planets
# confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']>0.0]
# no_confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']==0.0]
#
#
# #### TASK 2: Preparing the train and test data
# # Getting the list of Kepler ID
# confirmed_planets_KIC_list = confirmed_planet_df['kepid'].values
# no_confirmed_planets_KIC_list = no_confirmed_planet_df['kepid'].values
#
# # Shuffling the list of Kepler ID
# if(randomize_bool):
#     random.seed(SEED_VALUE)
# random.shuffle(confirmed_planets_KIC_list)
# random.shuffle(no_confirmed_planets_KIC_list)
#
# # CLASS 0 - NO CONFIRMED PLANETS
# # CLASS 1 - CONFIRMED PLANETS
# train_class_0_size = int(train_size * SPLIT_PERCENT)
# train_class_1_size = train_size - train_class_0_size
# print('Train',train_class_0_size,train_class_1_size)
#
# validation_class_0_size = int(validation_size * SPLIT_PERCENT)
# validation_class_1_size = validation_size - validation_class_0_size
# print('Validation',validation_class_0_size,validation_class_1_size)
#
# test_class_0_size = int(test_size * SPLIT_PERCENT)
# test_class_1_size = test_size - test_class_0_size
# print('Test',test_class_0_size,test_class_1_size)
#
# data_df = pd.DataFrame(columns=['KIC','Graph'])
# kic_list = list(confirmed_planets_KIC_list[0:int(train_class_1_size + validation_class_1_size + test_class_1_size)])
# kic_list.extend(list(no_confirmed_planets_KIC_list[0:int(train_class_0_size + validation_class_0_size + test_class_0_size)]))
#
# print("Plotting graphs for : "+str(len(kic_list))+' KIC ID')
#
# # Plotting the TCE for - Confirmed planets and No Confirmed Planets
# data_df['KIC'] = kic_list
#
# # Input image dimensions
# img_shape = image_array_fn(data_df['KIC'].values[0]).shape
# img_rows, img_cols = img_shape[0], img_shape[1]
#
# print('Image size = '+str(img_rows)+' by ' +str(img_cols))
#
# # Getting the KIC ID for train samples
# kic_train_list = list(confirmed_planets_KIC_list[0:int(train_class_1_size)])
# kic_train_list.extend(list(no_confirmed_planets_KIC_list[0:int(train_class_0_size)]))
# y_train = [1]*int(train_class_1_size)+[0]*int(train_class_0_size)
#
# # Getting the KIC ID for validation samples
# kic_val_list = list(confirmed_planets_KIC_list[int(train_class_1_size):int(train_class_1_size + validation_class_1_size)])
# kic_val_list.extend(list(no_confirmed_planets_KIC_list[int(train_class_0_size):
#                                                        int(train_class_0_size + validation_class_0_size)]))
# y_val = [1]*int(validation_class_1_size)+[0]*int(validation_class_0_size)
#
# # Getting the KIC ID for test samples
# kic_test_list = list(confirmed_planets_KIC_list[int(train_class_1_size + validation_class_1_size):
#                                                 int(train_class_1_size + validation_class_1_size + test_class_1_size)])
# kic_test_list.extend(list(no_confirmed_planets_KIC_list[int(train_class_0_size + validation_class_0_size):
#                                                         int(train_class_0_size + validation_class_0_size + test_class_0_size)]))
# y_test = [1]*int(len(list(confirmed_planets_KIC_list[int(train_class_1_size + validation_class_1_size):
#                                                 int(train_class_1_size + validation_class_1_size + test_class_1_size)])))+\
#          [0]*int(len(list(no_confirmed_planets_KIC_list[int(train_class_0_size + validation_class_0_size):
#                                                         int(train_class_0_size + validation_class_0_size + test_class_0_size)])))
# print('y_test_size = '+str(len(y_test)))
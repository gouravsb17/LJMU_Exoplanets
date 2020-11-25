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
from datetime import datetime

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, \
    MaxPooling2D, Lambda,Bidirectional,LSTM, Reshape

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve

os.chdir('..')

tqdm.pandas(desc="Progress: ")
warnings.filterwarnings('ignore')

DPI = 80
POINT_SIZE = 0.1
AXIS_LIMIT = 0.03
COLOR = 'b'

Y_LIM = {
           80: [40, 348],
           90: [42, 392],
          100: [50, 430],
          110: [55, 475],
          120: [65, 520],
          130: [72, 560],
          140: [79, 602]
        }

X_LIM = {
           80: [ 78, 445],
           90: [ 78, 508],
          100: [100, 560],
          110: [112, 615],
          120: [120, 669],
          130: [130, 722],
          140: [141, 778]
        }

SPLIT_PERCENT = 0.80
CROP_FLAG = True
randomize_bool = True
SEED_VALUE = 1

# Model Params
train_size, validation_size, test_size = 6000, 100, 1000
BATCH_SIZE = 32
PER_TRAIN_SAMPLE = 10
EPOCH = 20
VALIDATION_STEP = 5
THRESHOLD = 0.0001
CLASS_WEIGHT = {0:1.0,1:4.0}

# Paths
image_res_path = 'res/KIC_flux_graphs_'+str(DPI)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/'
model_save_path = 'res/model/'

def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)
    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)
    return lc_nan_normalize_flatten

def image_array_fn(kepler_id):
    try:
        img = cv2.imread(image_res_path + 'kepler_ID_'+ str(kepler_id) + '.png', 0)/ 255.0
        print(str(np.array(img).shape[0])+' X '+ str(np.array(img).shape[1]))
    except Exception as e:
        print(e,str(kepler_id))
    if(CROP_FLAG):
        return np.array(img[Y_LIM.get(DPI)[0]:Y_LIM.get(DPI)[1],
                            X_LIM.get(DPI)[0]:X_LIM.get(DPI)[1]])
    else:
        return img

for DPI in [80, 90, 100, 110, 120, 130, 140]:
    image_res_path = 'res/KIC_flux_graphs_' + str(DPI) + '_dpi_' + str(POINT_SIZE).split('.')[
        -1] + '_size_color_' + str(COLOR) + '/'
    print(DPI)
    for im in os.listdir(image_res_path):
        final_img = image_array_fn(im.split('_')[-1].split('.')[0])
        print(str(np.array(final_img).shape[0])+' X '+ str(np.array(final_img).shape[1]))
        break

exit()

#### TASK 1: Getting the train and test data
# Reading the data showing which KIC has planets and which not
complete_kepler_df = pd.read_csv('planetary_data/planetary_data_kepler_mission.csv',sep=',',dtype={'kepid':str})
complete_kepler_df = complete_kepler_df[['kepid','nconfp','nkoi']]

# Reading the metadata for which the data has been downloaded
metadata_df = pd.read_csv('planetary_data/kepler_download_metadata.csv',sep=',',dtype={'kic':str})
metadata_df = metadata_df.loc[(metadata_df['download']=='Y') & (metadata_df['fits_count']>=14)]

# Merging the data that has been downloaded and the plant info df
complete_kepler_df = pd.merge(complete_kepler_df,metadata_df,left_on='kepid',right_on='kic',how='inner')

# Separating the data into KIC having planets and not having any confirmed planets
confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']>0.0]
no_confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']==0.0]

#### TASK 2: Preparing the train, validation and test data using the stratified approach as we have limited computation and data
# Getting the list of Kepler ID
confirmed_planets_KIC_list = confirmed_planet_df['kepid'].values
no_confirmed_planets_KIC_list = no_confirmed_planet_df['kepid'].values

confirmed_planets_KIC_dict = {}
for kic in confirmed_planets_KIC_list:
  confirmed_planets_KIC_dict[kic] = 1

no_confirmed_planets_KIC_dict = {}
for kic in no_confirmed_planets_KIC_list:
  no_confirmed_planets_KIC_dict[kic] = 0

no_confirmed_planets_KIC_list = list(no_confirmed_planets_KIC_list)
no_confirmed_planets_KIC_list.remove('10000300')

print('Total KIC with confirmed planets vs non confirmed planets = ',len(confirmed_planets_KIC_list),'/',len(no_confirmed_planets_KIC_list))

# CLASS 0 - NO CONFIRMED PLANETS
# CLASS 1 - CONFIRMED PLANETS
train_class_0_size = int(train_size * SPLIT_PERCENT)
train_class_1_size = train_size - train_class_0_size

validation_class_0_size = int(validation_size * SPLIT_PERCENT)
validation_class_1_size = validation_size - validation_class_0_size

test_class_0_size = int(test_size * SPLIT_PERCENT)
test_class_1_size = test_size - test_class_0_size

# Getting the KIC ID for test samples
kic_test_list = list(confirmed_planets_KIC_list[0:int(test_class_1_size)])
kic_test_list.extend(list(no_confirmed_planets_KIC_list[0:int(test_class_0_size)]))

# Getting the KIC ID for validation samples
kic_val_list = list(confirmed_planets_KIC_list[int(test_class_1_size):
                                               int(test_class_1_size + validation_class_1_size)])
kic_val_list.extend(list(no_confirmed_planets_KIC_list[int(test_class_0_size):
                                                       int(test_class_0_size + validation_class_0_size)]))

# Getting the KIC ID for train samples
kic_train_list = list(confirmed_planets_KIC_list[int(test_class_1_size + validation_class_1_size):
                                                int(test_class_1_size + validation_class_1_size + train_class_1_size)])
kic_train_list.extend(list(no_confirmed_planets_KIC_list[int(test_class_0_size + validation_class_0_size):
                                                        int(test_class_0_size + validation_class_0_size + train_class_0_size)]))


random.seed(SEED_VALUE)
random.shuffle(kic_train_list)
random.shuffle(kic_val_list)
random.shuffle(kic_test_list)
kic_train_list = list(kic_train_list)
kic_val_list = list(kic_val_list)
kic_test_list = list(kic_test_list)

y_train = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_train_list]
y_val = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_val_list]
y_test = [confirmed_planets_KIC_dict.get(x,no_confirmed_planets_KIC_dict.get(x)) for x in kic_test_list]

print(len(kic_train_list),"Train Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_train_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_train_list]))

print(len(kic_val_list),"Validation Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_val_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_val_list]))

print(len(kic_test_list),"Test Split (Confirmed Planet/No planets) = ",sum([confirmed_planets_KIC_dict.get(x,0)==1 for x in kic_test_list]),"/",
      sum([no_confirmed_planets_KIC_dict.get(x,1)==0 for x in kic_test_list]))

# Input image dimensions
img_shape = image_array_fn(kic_train_list[0]).shape
img_rows, img_cols = img_shape[0], img_shape[1]

print('Image size = '+str(img_rows)+' by ' +str(img_cols))

# Generator Function
def generator(samples, batch_size = BATCH_SIZE):
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
                image = image_array_fn(batch_sample[0])
                # Read label (y)
                y = batch_sample[1]
                # Add example to arrays
                X_train.append(image)
                y_train.append(float(y))

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

            # The generator-y part: yield the next training batch
            yield X_train, y_train

# Import list of train and validation data (image filenames and image labels)
train_samples = np.array(list(map(lambda x,y: [x,y], kic_train_list,y_train)))
validation_samples = np.array(list(map(lambda x,y: [x,y], kic_val_list,y_val)))

# Create generator
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

###LSTM CNN
# def ReshapeLayer(x):
#     shape = x.shape
#
#     # 1 possibility: H,W*channel
#     reshape = Reshape((shape[1], shape[2] * shape[3]))(x)
#
#     # 2 possibility: W,H*channel
#     # transpose = Permute((2,1,3))(x)
#     # reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)
#
#     return reshape
#
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Conv2D(filters=24, kernel_size=(4, 4), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# model.add(Lambda(ReshapeLayer))
# model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=False)))
# model.add(Dense(1, activation='sigmoid'))

### CNN Model

# model = Sequential()
# model.add(Conv2D(4, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(img_rows,img_cols,1)))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# print(model.summary())
#
# # define the checkpoint
# filepath = model_save_path+"model.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
#
# # Fit model using generator
# model.fit(train_generator,
#           steps_per_epoch=len(train_samples)//PER_TRAIN_SAMPLE,
#           epochs=EPOCH,
#           validation_data=validation_generator,
#           validation_steps=VALIDATION_STEP,
#           class_weight=CLASS_WEIGHT,
#           callbacks=callbacks_list
#           )
#
# date_time_now = str(datetime.now()).split('.')[0]
#
# model_json = model.to_json()
# if(CROP_FLAG):
#     model_name = 'model-'+str(DPI)+'-dpi-cropped-'+str(date_time_now)
# else:
#     model_name = 'model-' + str(DPI) + '-dpi-'+str(date_time_now)
#
# with open(model_save_path + model_name+".json", "w") as json_file:
#     json_file.write(model_json)
#
# model.save_weights(model_save_path + model_name+".h5")
# print("Saved model to disk")

model_name = 'model-130-dpi-cropped-2020-10-31 23:39:55'

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
test_X_samples = np.array(list(map(lambda x: image_array_fn(x),kic_test_list)))
test_X_samples = test_X_samples.reshape(test_X_samples.shape[0],test_X_samples.shape[1],test_X_samples.shape[2],1)
print(test_X_samples.shape)

predit_output = loaded_model.predict(test_X_samples)

print(list(predit_output))

print(model_name)

precision_list = []
recall_list = []
f_measure_list = []

tn_list = []
fp_list = []
fn_list = []
tp_list = []

for THRESHOLD in list([x for x in [0.0001, 0.001, 0.5]]):
    y_pred = list(map(lambda x: 0 + 1 * (x >= THRESHOLD), predit_output))
    y_true = y_test


    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


    print('Test Results: - Threeshold - ' + str(THRESHOLD))
    print('True Negative (No Confirmed Planets) = ' + str(tn(y_true, y_pred)),
          'False Positive = ' + str(fp(y_true, y_pred)),
          'False Negative = ' + str(fn(y_true, y_pred)),
          'True Positive (Confirmed Planets) = ' + str(tp(y_true, y_pred)))
    tn_list.append(str(tn(y_true, y_pred)))
    fp_list.append(str(fp(y_true, y_pred)))
    fn_list.append(str(fn(y_true, y_pred)))
    tp_list.append(str(tp(y_true, y_pred)))

    # calculate prediction
    precision = precision_score(y_true, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    precision_list.append(precision)

    # calculate recall
    recall = recall_score(y_true, y_pred, average='binary')
    print('Recall: %.3f' % recall)
    recall_list.append(recall)

    # calculate f1 score
    score = f1_score(y_true, y_pred, average='binary')
    print('F-Measure: %.3f' % score)
    f_measure_list.append(score)

print(' / '.join([str(np.round(x, 3)) for x in precision_list]))
print(' / '.join([str(np.round(x, 3)) for x in recall_list]))
print(' / '.join([str(np.round(x, 3)) for x in f_measure_list]))

print(' / '.join([str(x) for x in tn_list]))
print(' / '.join([str(x) for x in fp_list]))
print(' / '.join([str(x) for x in fn_list]))
print(' / '.join([str(x) for x in tp_list]))

THRESHOLD = 0.0001
y_pred = list(map(lambda x: 0 + 1 * (x >= THRESHOLD), predit_output))
y_true = y_test

from sklearn.metrics import precision_recall_curve
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

# ROC curve and AUC on an imbalanced dataset
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

print('Complete!')
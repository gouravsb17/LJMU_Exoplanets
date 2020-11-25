# Importing the required libraries
import os
import cv2
import timeit
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import lightkurve as lk
import matplotlib.pyplot as plt

os.chdir('..')

tqdm.pandas(desc="Progress: ")
warnings.filterwarnings('ignore')

DPI = 140
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
          80: [78, 445],
          90: [78, 508],
         100: [100, 560],
         110: [112, 615],
         120: [120, 669],
         130: [130, 722],
         140: [141, 778]
         }

SPLIT_PERCENT = 0.75
CROP_FLAG = True
randomize_bool = True

# Paths
image_res_path = 'res/KIC_flux_graphs_'+str(DPI)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/'
kepler_id = '12207117'
img = cv2.imread(image_res_path + 'kepler_ID_'+ str(kepler_id) + '.png', 0)/ 255.0
cropped_img = np.array(img[Y_LIM.get(DPI)[0]:Y_LIM.get(DPI)[1],
                            X_LIM.get(DPI)[0]:X_LIM.get(DPI)[1]])

(h, w) = cropped_img.shape[:2]
# calculate the center of the image
center = (w / 2, h / 2)
angle90 = 90
scale = 1.0

# Perform the counter clockwise rotation holding at the center
# 90 degrees
M = cv2.getRotationMatrix2D(center, angle90, scale)
cropped_img_90_rotated = cv2.warpAffine(cropped_img, M, (h, w))

print(cropped_img_90_rotated.shape)
cv2.imshow('cropped image rotated 90', cropped_img_90_rotated)

print(img.shape)
cv2.imshow('cropped image', cropped_img)

cv2.imshow('image', img)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
print(cropped_img.shape)
exit()
model_save_path = 'res/model/'

def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)
    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)
    return lc_nan_normalize_flatten

def image_array_fn(kepler_id):
    img = cv2.imread(image_res_path + 'kepler_ID_'+ str(kepler_id) + '.png', 0)/ 255.0
    if(CROP_FLAG):
        return np.array(img[Y_LIM.get(DPI)[0]:Y_LIM.get(DPI)[1],
                            X_LIM.get(DPI)[0]:X_LIM.get(DPI)[1]])
    else:
        return img

def generate_TCE(kepler_id,
                 point_size=POINT_SIZE,
                 dpi_val=DPI,
                 axis_limit=AXIS_LIMIT):
    kepler_id = str(kepler_id).strip(' ')

    # if('kepler_ID_'+str(kepler_id)+'.png' in os.listdir(image_res_path)):
    #     return 'Graph already present'

    # Pulling from the External HDD to the temp resource folder
    try:
        res_path = '/Volumes/PaligraphyS/kepler_data/res/kepler_ID_'+kepler_id+'/'
        shutil.copytree(res_path,'temp_res/kepler_ID_'+kepler_id+'/')
        res_path = 'temp_res/kepler_ID_'+kepler_id+'/'
    except:
        print('Error here')
        return 'Data for KIC - '+ kepler_id +' not in Ext HDD'

    lc_list_files = []
    for lc_file in os.listdir(res_path):
        if('llc.fits' in lc_file):
            lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile(res_path+lc_file))

    lc_collection = lk.LightCurveFileCollection(lc_list_files)
    stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

    corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
    corrected_lc_df = corrected_lc.to_pandas()
    corrected_lc_df['flux'] = corrected_lc_df['flux']-1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(corrected_lc_df['time'],corrected_lc_df['flux'],s=point_size)
    ax.set_ylim(-1*axis_limit, axis_limit)

    plt.axis('off')
    if('kepler_ID_'+ kepler_id + '.png' not in os.listdir('res/KIC_flux_graphs_'+str(80)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/')):
        plt.savefig('res/KIC_flux_graphs_'+str(80)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=80)
    plt.savefig('res/KIC_flux_graphs_'+str(90)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=90)
    plt.savefig('res/KIC_flux_graphs_'+str(100)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=100)
    plt.savefig('res/KIC_flux_graphs_'+str(110)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=110)
    plt.savefig('res/KIC_flux_graphs_'+str(130)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=130)
    plt.savefig('res/KIC_flux_graphs_'+str(140)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/' +'kepler_ID_'+ kepler_id, dpi=140)

    # Removing the kepler data brought to the temporary directory
    shutil.rmtree('temp_res/kepler_ID_'+kepler_id)

    plt.close()
    print('Graph plotted')
    return 'Graph plotted'

def multiprocessor_plot_fn(data_df,generate_TCE_fn,dpi_val=DPI,point_size=POINT_SIZE):

    temp_dir = "res/tempDir"

    KIC_list = []
    for iter, rows in data_df.iterrows():
        KIC_list.append((rows['KIC']))

    cores = multiprocessing.cpu_count()

    def worker(sub_trackers_list, start):
        plotted_df = pd.DataFrame(columns=['KIC','Graph'])
        k = start
        for sub_tracker in tqdm(sub_trackers_list):
            plotted_df.loc[k] = [sub_tracker, generate_TCE_fn(sub_tracker,point_size,dpi_val)]
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

    final_KIC_plotted_df = pd.DataFrame(columns=['KIC','Graph'])
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

# Merging the data that has been downloaded and the plant info df
complete_kepler_df = pd.merge(complete_kepler_df,metadata_df,left_on='kepid',right_on='kic',how='inner')

# Separating the data into KIC having planets and not having any confirmed planets
confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']>0.0]
no_confirmed_planet_df = complete_kepler_df.loc[complete_kepler_df['nconfp']==0.0]

KIC_images_generated = []
for png_file in os.listdir('res/KIC_flux_graphs_'+str(80)+'_dpi_'+str(POINT_SIZE).split('.')[-1]+'_size_color_'+str(COLOR)+'/'):
    if('.png' in png_file):
        KIC_images_generated.append(png_file.split('.png')[0].split('_')[-1])

KIC_images_to_be_generated = []
for kic in complete_kepler_df['kic'].values:
    if(kic not in KIC_images_generated):
        KIC_images_to_be_generated.append(kic)

print("Number of KIC ID for which images needs generation = "+str(len(KIC_images_to_be_generated)))

data_df = pd.DataFrame()
data_df['KIC'] = KIC_images_to_be_generated

# KIC_images_to_be_generated
multiprocessor_plot_fn(data_df,generate_TCE)



# https://stripe.com/en-ca




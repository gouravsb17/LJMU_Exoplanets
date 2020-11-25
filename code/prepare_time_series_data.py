# Importing the required libraries
import os
import timeit
import shutil
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import lightkurve as lk
from numpy import save

os.chdir('..')

tqdm.pandas(desc="Progress: ")
warnings.filterwarnings('ignore')

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

# Getting the KIC ID's for which numpy time series array needs to be generated
new_TCE_Array = []
for kic in metadata_df['kic'].values:
    if(kic not in KIC_array_list):
        new_TCE_Array.append(kic)

print("Number of new time series arrays to be genereated : "+str(len(new_TCE_Array)))

data_df = pd.DataFrame(columns=['KIC'])
data_df['KIC'] = new_TCE_Array
multiprocessor_plot_fn(data_df,generate_time_series_fn)

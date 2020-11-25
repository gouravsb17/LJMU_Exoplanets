# Importing the required libraries
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
import os,shutil
import numpy as np
from astropy.time import Time
from lightkurve.correctors import KeplerCBVCorrector
from sklearn import preprocessing

os.chdir('..')

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)

kepler_data_ingested = pd.read_csv('planetary_data/kepler_download_metadata.csv',sep=',',dtype=str)
kepler_data_ingested['fits_count'] = kepler_data_ingested['fits_count'].astype(int)
kepler_data_ingested = kepler_data_ingested.sort_values('fits_count',ascending=False)

# print(kepler_data_ingested.groupby('fits_count').count())
#
# kepler_data_ingested = kepler_data_ingested.loc[kepler_data_ingested['fits_count']>=14]
#
# # kepler_data_ingested = kepler_data_ingested.loc[kepler_data_ingested['fits_count']==34]
#
# print(kepler_data_ingested.groupby('fits_count').count())
#
# exit()

def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)

    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)

    lc_flat, trend_lc = lc_nan_normalize_flatten.flatten(return_trend=True)
    return lc_flat

for kepler_id in list(kepler_data_ingested['kic']):
    # Kepler ID whose data needs to be analysed
    kepler_id = '10028792'
    # if('kepler_ID_'+kepler_id+'.png' in os.listdir('KIC_flux_graphs_temp/')):
    #     continue

    res_path = 'res/kepler_ID_' + kepler_id + '/'
    try:
        # Getting from local if already present
        os.listdir(res_path)
    except:
        try:
            # Pulling from the External HDD to the temp resource folder
            res_path = '/Volumes/PaligraphyS/kepler_data/res/kepler_ID_'+kepler_id+'/'
            shutil.copytree(res_path,'temp_res/kepler_ID_'+kepler_id+'/')
            res_path = 'temp_res/kepler_ID_'+kepler_id+'/'
        except:
            if(kepler_id not in list(kepler_data_ingested['kic'])):
                print('Data for KIC not downloaded')
                exit()
            else:
                res_path = 'temp_res/kepler_ID_' + kepler_id + '/'

    lc_list_files = []
    for lc_file in os.listdir(res_path):
        if('llc.fits' in lc_file):
            lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile(res_path+lc_file))

    lc_collection = lk.LightCurveFileCollection(lc_list_files)
    stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

    corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
    corrected_lc_df = corrected_lc.to_pandas()
    corrected_lc_df['flux'] = corrected_lc_df['flux']-1

    # positive_median_val = 2.0*np.median(corrected_lc_df['flux'][corrected_lc_df['flux']>0])
    # negative_median_val = 2.0*np.median(corrected_lc_df['flux'][corrected_lc_df['flux']<0])
    #
    # corrected_lc_df['flux'] = corrected_lc_df['flux'].apply(lambda x:
    #                                     x-positive_median_val if(x>0) else x-negative_median_val)

    # corrected_lc_df['flux'] = preprocessing.scale(np.array(corrected_lc_df['flux']).reshape(-1,1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(corrected_lc_df['time'],corrected_lc_df['flux'],s=0.1)
    ax.set_ylim(-0.025, 0.025)

    plt.show()
    break

    plt.savefig('KIC_flux_graphs_temp/unclear_kepler_ID_'+ kepler_id,dpi=500)

    # Removing the kepler data brought to the temporary directory
    shutil.rmtree('temp_res/kepler_ID_'+kepler_id)

    plt.close()

    exit()

exit()


# EXPERIMENTAL STUFF:

# from lightkurve import search_lightcurvefile
# import matplotlib.pyplot as plt
#
# lcf =search_lightcurvefile('8197788').download()
# lcf.PDCSAP_FLUX.plot()
# plt.show()
#
# exit()

# corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
# corrected_lc_df = corrected_lc.to_pandas()
#
# plt.plot(corrected_lc_df['time'],corrected_lc_df['flux'])
# plt.savefig('fig1',dpi=200)
# plt.show()

# from statsmodels.tsa.seasonal import seasonal_decompose
# corrected_lc_df = corrected_lc.to_pandas()
# corrected_lc_series = corrected_lc_df['flux']
# result = seasonal_decompose(corrected_lc_series, model='additive')
# result.plot()
# plt.show()


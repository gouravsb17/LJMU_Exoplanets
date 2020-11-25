from astropy.io import fits
import os
import pandas as pd
from astropy.table import Table
import lightkurve as lk
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir('..')

# CONFIG Variables
kepler_id = 'kepler_ID_757450'

def fits_data_to_pandas_df_fn(filename):

    # Extracting the basic info present in the file and displaying it
    with fits.open(filename) as hdul:
        print(hdul.info())

    # Extacting the table from the fits file
    dat = Table.read(filename, format='fits')
    table_df = dat.to_pandas()

    return(table_df)

def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)

    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)

    lc_flat, trend_lc = lc_nan_normalize_flatten.flatten(return_trend=True)
    return lc_flat

table_df = pd.DataFrame()

for file_name in os.listdir('temp_res/'+kepler_id+'/'):
    if('llc.fits' in file_name):
        try:
            fits_filename = 'temp_res/'+kepler_id+'/'+file_name
            table_df = pd.concat([table_df,fits_data_to_pandas_df_fn(fits_filename)])
        except:
            continue

lc_list_files = []
for lc_file in os.listdir('temp_res/'+kepler_id+'/'):
    if ('llc.fits' in lc_file):
        print(lc_file)
        lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile('temp_res/'+kepler_id+'/' + lc_file))

lc_collection = lk.LightCurveFileCollection(lc_list_files)
stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

transformed_values = my_custom_corrector_func(stitched_lc_PDCSAP)
transformed_values = transformed_values.to_pandas()

print(table_df)
print(list(table_df))
print(transformed_values)
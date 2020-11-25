# Importing the required libraries
import os
import random
import warnings
from tqdm import tqdm
import lightkurve as lk
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)


warnings.filterwarnings('ignore')
tqdm.pandas(desc="Progress: ")

# Changing the directory
os.chdir('..')

# dpi_simulated_images_df = pd.DataFrame(columns=['Injection Group','DPI','Total Simulated Images'])
# k=0
# for inj in [1, 2, 3]:
#     for dpi in [80,90,100,110,120,130,140]:
#         try:
#             dpi_simulated_images_df.loc[k] = [dpi,inj,
#                   len(os.listdir('simulated_images_data/simulated_inj'+str(inj)+'_graph_'+str(dpi)+'_dpi'))]
#             k+=1
#         except:
#             pass
# print(dpi_simulated_images_df)
# exit()


# Config variables
INJECTION = 3
DPI_list = [80, 90, 100, 110, 120, 130, 140]
SEED_VALUE = 100
POINT_SIZE = 0.1
AXIS_LIMIT = 0.03
COLOR = 'b'
NUMBER_OF_IMAGES_DICT = {1: 1463, 2: 340, 3: 200}
GET_DATA = False #True

def worker(sub_kic_list):
    if(len(sub_kic_list)==0):
        return
    else:
        for sub_kic_line in tqdm(sub_kic_list):
            os.system('wget -O simulated_data/Injected_Group_' + str(INJECTION) + '/' + sub_kic_line[8:])
        return

def multiprocessor_fn(complete_kic_list):

    cores = multiprocessing.cpu_count()
    slot = int(np.ceil(len(complete_kic_list) / cores))
    end = 0

    complete_sub_kic_list = []

    for i in range(cores):
        start = end
        if (start == slot * (cores - 1)):
            end = len(complete_kic_list)
        else:
            end = start + slot
        complete_sub_kic_list.append(complete_kic_list[start:end])

    pool = multiprocessing.Pool(processes=4)
    pool.map(worker, complete_sub_kic_list)

for DPI in DPI_list:
    # File Paths
    new_dir = 'simulated_images_data/simulated_inj' + str(INJECTION) + '_graph_' + str(DPI) + '_dpi/'

    # Reading the respective injected bat file
    bat_file = 'simulated_data/injected-light-curves-dr25-inj' + str(INJECTION) + '.bat'
    file1 = open(bat_file, 'r')
    lines = file1.readlines()

    # Getting the KIC ID DICT from the respective bat file
    # Key - KPLR ID, Value - number of fits file available
    kplr_ids_dict = {}
    for line in lines:
        if ('kplr' in line):
            kplr_ids_dict[line.split('-O ')[-1].split('-')[0]] = kplr_ids_dict.get(line.split('-O ')[-1].split('-')[0],
                                                                                   0) + 1
    print('kplr_ids_dict',len(kplr_ids_dict))

    # Getting the KIC ID's LIST for which number of fits file is greater than 14
    kplr_id_list = []
    for key, value in kplr_ids_dict.items():
        if (value >= 14):
            kplr_id_list.append(key)
    print('kplr_id_list', len(kplr_id_list))

    # Shuffling the KPLR ID before making the selection
    random.seed(SEED_VALUE)
    random.shuffle(kplr_id_list)

    # Selecting the top KIC ID's to generate the images of their graph
    kplr_id_list = list(kplr_id_list)[0:NUMBER_OF_IMAGES_DICT[INJECTION]]
    print('kplr_id_list', len(kplr_id_list))

    # Getting the KPLR ID's for which fits files are already present
    try:
        llc_fits_files = os.listdir('simulated_data/Injected_Group_' + str(INJECTION) + '/')
        llc_fits_files = list(filter(lambda x: 'INJECTED-inj' + str(INJECTION) + '_llc.fits' in x, llc_fits_files))
        kic_generated_list = [x.split('-')[0] for x in llc_fits_files]
    except:
        kic_generated_list = []

    kic_generated_list = list(set(kic_generated_list))
    print('kic_generated_list', len(kic_generated_list))

    # Removing the kic whose fits file is already present
    required_kplr_list = []
    for kic_id in kplr_id_list:
        if (kic_id not in kic_generated_list):
            required_kplr_list.append(kic_id)
    required_kplr_list = required_kplr_list[0:len(kplr_id_list)-len(kic_generated_list)]
    print('required_kplr_list', len(required_kplr_list))

    # Getting the required fits files from the source
    if(GET_DATA):
        complete_kic_line_list = []
        for kic_id in required_kplr_list:
            if (kic_id in kic_generated_list):
                continue
            for line in lines:
                if (kic_id in line and 'inj' + str(INJECTION) + '_llc' in line):
                    complete_kic_line_list.append(line)

        print('complete_kic_line_list',len(complete_kic_line_list))

        multiprocessor_fn(complete_kic_line_list)

        os.chdir('simulated_data/Injected_Group_' + str(INJECTION) + '/')
        os.system('gunzip *.gz;rm -rf __MACOSX')
        os.chdir('../..')


    # Custom corrector function used in pre-processing steps for the training data
    def my_custom_corrector_func(lc_raw):
        # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
        # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
        lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)

        lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)

        lc_flat, trend_lc = lc_nan_normalize_flatten.flatten(return_trend=True)
        return lc_flat


    # Creating the new directory to store the images
    try:
        os.mkdir(new_dir)
    except:
        print('Directory already made...')

    # Getting the llc files from which images will be generated
    llc_files = os.listdir('simulated_data/Injected_Group_' + str(INJECTION) + '/')
    llc_files = list(filter(lambda x: '.fits' in x and 'inj' + str(INJECTION) in x, llc_files))

    # Getting the KPLR ID for which images already exists
    try:
        png_files = [x.split('_')[0] for x in os.listdir(new_dir)]
    except:
        png_files = []

    # Generating the images for the Injected Group selected
    for kic_id in tqdm(kplr_id_list):
        kic_files = list(filter(lambda x: kic_id in x, llc_files))

        if (len(kic_files) < 14 or kic_id in png_files):
            continue

        lc_list_files = []
        for llc_file in kic_files:
            try:
                lc_list_files.append(lk.lightcurvefile.
                                     KeplerLightCurveFile(
                    'simulated_data/Injected_Group_' + str(INJECTION) + '/' + llc_file))
            except:
                print('Error:' + llc_file)
                continue

        lc_collection = lk.LightCurveFileCollection(lc_list_files)
        stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

        corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
        corrected_lc_df = corrected_lc.to_pandas()
        corrected_lc_df['flux'] = corrected_lc_df['flux'] - 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(corrected_lc_df['time'], corrected_lc_df['flux'], s=POINT_SIZE, c=COLOR)
        ax.set_ylim(-1 * AXIS_LIMIT, AXIS_LIMIT)
        ax.set_axis_off()
        # plt.show()
        plt.savefig(new_dir + kic_id + '_INJECTED_' + str(INJECTION) + '.png',
                    dpi=DPI)

        plt.close()

# 18775859427

# column_names= 'KIC_ID  | Sky_Group | i_period |  i_epoch  |  N_Transit| i_depth |   i_dur    |  i_b   | i_ror   |  i_dor   | EB_injection | Offset_from_source | Offset_distance | Expected_MES | Recovered |    TCE_ID    | Measured_MES | r_period |    r_epoch   | r_depth |   r_dur     |   r_b   | r_ror   |  r_dor   | Fit_Provenance'
# column_names=[x.strip(' ') for x in column_names.split('|')]
# df = pd.read_table('planetary_data/simulated_metadata_INJ2',skiprows=73,names=column_names,sep='\s+')
# df = df[['KIC_ID','EB_injection','Offset_from_source']]
#
# print(df.groupby('EB_injection').count()['KIC_ID'])
# print(df.groupby('Offset_from_source').count()['KIC_ID'])

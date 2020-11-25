# Importing the required libraries
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
import os, shutil
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from tqdm import tqdm
import warnings
import seaborn as sns

os.chdir('..')
tqdm.pandas(desc="Progress: ")
warnings.filterwarnings('ignore')

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)


def my_custom_corrector_func(lc_raw):
    # Source: https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
    # Clean outliers, but only those that are above the mean level (e.g. attributable to stellar flares or cosmic rays).
    lc_clean_outliers = lc_raw.remove_outliers(sigma=20, sigma_upper=4)

    lc_nan_normalize_flatten = lc_clean_outliers.remove_nans().normalize().flatten(window_length=101)

    lc_flat, trend_lc = lc_nan_normalize_flatten.flatten(return_trend=True)
    return lc_flat


def read_kepler_data_from_external_HDD(kepler_id):
    res_path = 'res/kepler_ID_' + kepler_id + '/'
    try:
        # Getting from local if already present
        os.listdir(res_path)
    except:
        try:
            # Pulling from the External HDD to the temp resource folder
            res_path = '/Volumes/PaligraphyS/kepler_data/res/kepler_ID_' + kepler_id + '/'
            shutil.copytree(res_path, 'temp_res/kepler_ID_' + kepler_id + '/')
            res_path = 'temp_res/kepler_ID_' + kepler_id + '/'
        except Exception as e:
            if ('File exists: ' in str(e)):
                res_path = 'temp_res/kepler_ID_' + kepler_id + '/'
            else:
                print('Data for KIC not downloaded')
                return [False, np.array([])]

    lc_list_files = []
    for lc_file in os.listdir(res_path):
        if ('llc.fits' in lc_file):
            lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile(res_path + lc_file))

    lc_collection = lk.LightCurveFileCollection(lc_list_files)
    stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

    corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
    corrected_lc_df = corrected_lc.to_pandas()
    corrected_lc_df['flux'] = corrected_lc_df['flux'] - 1

    # Removing the kepler data brought to the temporary directory
    shutil.rmtree('temp_res/kepler_ID_' + kepler_id)

    return [True, np.array([corrected_lc_df['time'], corrected_lc_df['flux']])]


try:
    stats_df = pd.read_csv('planetary_data/stats_df.csv', dtype={'KIC': str})
except:
    stats_df = pd.DataFrame(columns=['KIC', 'flux_point_counts', 'max_flux_value', 'min_flux_value',
                                     'avg_flux_value', 'median_flux_value', 'skewness_flux_value',
                                     'kurtosis_flux_value', 'Q1_flux_value', 'Q3_flux_value', 'std_flux_value',
                                     'variance_flux_value'])

# Getting the kepler ID's for which we will train and test the model
i = len(stats_df)
# for file in tqdm(os.listdir('res/KIC_flux_graphs_80_dpi_1_size_color_b/')):
#     if ('.png' in file):
#         kepler_id = file.split('_')[-1].split('.')[0]
#         if (kepler_id in list(stats_df['KIC'])):
#             continue
#         try:
#             response_list = read_kepler_data_from_external_HDD(kepler_id)
#         except:
#             print('Error in '+str(kepler_id))
#             continue
#         if (response_list[0]):
#             stats_df.loc[i] = [str(kepler_id), response_list[1].shape[1], np.max(response_list[1][1]),
#                                np.min(response_list[1][1]), np.average(response_list[1][1]),
#                                np.nanmedian(response_list[1][1]), skew(response_list[1][1]),
#                                kurtosis(response_list[1][1]), np.nanquantile(response_list[1][1], 0.25),
#                                np.nanquantile(response_list[1][1], 0.75),np.nanstd(response_list[1][1]),
#                                np.nanvar(response_list[1][1])]
#         i += 1
#
#     if (i % 20 == 0):
#         stats_df.drop_duplicates('KIC', inplace=True)
#         stats_df.to_csv('planetary_data/stats_df.csv', sep=',', index=False)

# exit()

complete_kepler_df = pd.read_csv('planetary_data/planetary_data_kepler_mission.csv', sep=',', dtype={'kepid': str})
complete_kepler_df = complete_kepler_df[['kepid', 'nconfp', 'nkoi']]

stats_planets_df = pd.merge(stats_df, complete_kepler_df, left_on='KIC', right_on='kepid')

stats_planets_df.drop_duplicates('KIC', inplace=True)
stats_planets_df.drop('kepid', inplace=True, axis=1)

stats_planets_df.to_csv('planetary_data/stats_planets_df.csv', sep=',', index=False)

stats_planets_df = stats_planets_df.loc[((stats_planets_df['max_flux_value']<=0.03) &
                                        (stats_planets_df['min_flux_value']>=-0.03)) |
                                        (stats_planets_df['nconfp']>0.0)]

stats_planets_df['Confirmed_planets'] = [1.0 * x for x in stats_planets_df['nconfp'] > 0.0]

print(stats_planets_df.groupby('Confirmed_planets').count()[['KIC']])

print(stats_planets_df.groupby(['Confirmed_planets', 'nkoi']).count()['KIC'])

print(stats_planets_df.loc[(stats_planets_df['nkoi'] == 0) &
                           (stats_planets_df['Confirmed_planets'] == 1)].sort_values('nkoi')[
          ['KIC', 'nkoi', 'Confirmed_planets']])


def plot_curve(x_column, y_column, hue_column="Confirmed_planets"):
    graph_name = y_column + '.png'

    if (x_column == 'nkoi'):
        x_label = 'Number of Kepler object of interest'
    else:
        x_label = x_column[0].upper() + x_column[1:].replace('_', ' ')

    y_label = y_column[0].upper() + y_column[1:].replace('_', ' ')
    # Plot 1: This will show the flux point counts for both the classes
    sns.set_theme(style="darkgrid")
    g = sns.catplot(x=x_column, y=y_column,
                    hue=hue_column,
                    data=stats_planets_df, kind="strip",
                    dodge=True,
                    height=4, aspect=1.5, legend_out=False)
    g.despine(left=True)
    # title
    new_title = hue_column.replace('_', ' ')
    g._legend.set_title(new_title)
    # replace labels
    new_labels = ['0 - No exoplanet', '1 - Exoplanet Present']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    g.set(xlabel=x_label, ylabel=y_label)
    plt.xlim(-0.5, 7.5)
    plt.tight_layout()
    plt.savefig('EDA_images/' + graph_name)
    # plt.show()
    plt.close()


y_columns = ['flux_point_counts', 'max_flux_value', 'min_flux_value',
             'avg_flux_value', 'median_flux_value', 'skewness_flux_value',
             'kurtosis_flux_value', 'Q1_flux_value', 'Q3_flux_value',
             'std_flux_value', 'variance_flux_value']
for y_column in y_columns:
    plot_curve('nkoi', y_column)

print(len(stats_planets_df.loc[stats_planets_df['nconfp'] > 0.0]))
print(len(stats_planets_df.loc[stats_planets_df['nconfp'] == 0.0]))

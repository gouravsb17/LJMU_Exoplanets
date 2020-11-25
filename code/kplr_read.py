import kplr
import os
import shutil
import logging
import pandas as pd
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)

logger = logging.getLogger('keplerTrack')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename='error_logs.log',mode='a')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

client = kplr.API(data_root='/Users/gouravsb/pycharmProjects/exoplanets_search_algo/res')

os.chdir('..')

def fetch_kepler_data(kic_id):
    logger.info("Getting Kepler data for KIC ID - " + str(kic_id))
    source = '/Users/gouravsb/pycharmProjects/exoplanets_search_algo/res/data/lightcurves/' + '0' * (
            9 - len(str(kic_id))) + str(kic_id)
    destination = '/Users/gouravsb/pycharmProjects/exoplanets_search_algo/res/kepler_ID_' + str(kic_id)

    try:
        star = client.star(kic_id)
        lightcurves = star.get_light_curves(fetch=True)
    except Exception as e:
        logger.error(str(e) + ' Error in fetching data from API for KIC - ' + str(kic_id))
        return

    for fits_file in os.listdir(source):
        try:
            shutil.move(source + '/' + fits_file, destination)
        except Exception as e:
            logger.error(str(e) + ' Error in moving the directory')

    os.rmdir(source)
    logger.info('FITS file count - ' + str(len(os.listdir(destination))))
    logger.info('Data fetched.')

try:
    kepler_download_df = pd.read_csv('planetary_data/kepler_download_metadata.csv')
except Exception as e:
    kepler_download_df = pd.DataFrame(columns=['kic','fits_count','download'])
    i=0
    for folder in os.listdir('res'):
        if('kepler_ID_' in folder):
            num_fits_file = np.sum([1 if('llc.fits' in x) else 0 for x in os.listdir('res/'+folder)])
            print(folder,num_fits_file)
            if(num_fits_file>0):
                kepler_download_df.loc[i] = [str(folder).split('_ID_')[-1],int(num_fits_file),'Y']
            else:
                kepler_download_df.loc[i] = [str(folder).split('_ID_')[-1],int(num_fits_file), 'N']
            i+=1
    kepler_download_df.to_csv('planetary_data/kepler_download_metadata.csv',sep=',',index=False)

kepler_download_df['ingested'] = 'Y'

kepler_download_df['kic_str'] = kepler_download_df['kic'].apply(lambda x: str(x))

complete_kepler_df = pd.read_csv('planetary_data/planetary_data_kepler_mission.csv',sep=',')

complete_kepler_df = complete_kepler_df[['kepid','nkoi','ntce','nconfp','teff']]

print(complete_kepler_df.groupby(['nkoi']).count().reset_index()[['nkoi','kepid']])

print(complete_kepler_df.groupby(['nconfp']).count().reset_index()[['nconfp','kepid']])

print(complete_kepler_df.groupby(['ntce']).count().reset_index()[['ntce','kepid']])

def convert_str_float(x):
    try:
        return float(x)
    except:
        print(x)
        return 0.0
complete_kepler_df['teff'] = complete_kepler_df['teff'].apply(convert_str_float)
complete_kepler_df = complete_kepler_df.loc[(complete_kepler_df['teff']!=0.0) &
                                            (complete_kepler_df['nkoi']==0.0) &
                                            (complete_kepler_df['nconfp']==0.0) &
                                            (complete_kepler_df['ntce']==0.0)]
k = pd.DataFrame(pd.cut(complete_kepler_df['teff'],bins=5))
k['counter'] = 1
print(k.groupby('teff').count())

exit()

complete_kepler_df['kepid_str'] = complete_kepler_df['kepid'].apply(lambda x: str(x))

complete_kepler_df = pd.merge(complete_kepler_df,kepler_download_df,left_on='kepid_str',
                              right_on='kic_str',how='left')
complete_kepler_df = complete_kepler_df.loc[complete_kepler_df['ingested']!='Y']

complete_kepler_df = complete_kepler_df[['kepid','nkoi','ntce']]

complete_kepler_df['ntce'] = complete_kepler_df['ntce'].apply(lambda x: float(str(x).strip(' ').replace('.0','')))

kepler_download_df.drop(['ingested','kic_str'],axis=1,inplace=True)

i=len(kepler_download_df)
for kic_id in complete_kepler_df.sort_values('ntce',ascending=False)['kepid']:
    try:
        os.mkdir('res/kepler_ID_' + str(kic_id))
        fetch_kepler_data(kic_id)
    except:
        if (len(os.listdir('res/kepler_ID_' + str(kic_id))) == 0):
            fetch_kepler_data(kic_id)

    if(len(os.listdir('res/kepler_ID_' + str(kic_id)))>0):
        kepler_download_df.loc[i] = [str(kic_id),np.sum([1 if('llc.fits' in x) else 0 for x in os.listdir('res/kepler_ID_' + str(kic_id))]),'Y']
    else:
        kepler_download_df.loc[i] = [str(kic_id), 0, 'N']

    if(i%10==0):
        print('Number of KIC downloaded = '+str(i))
        kepler_download_df.to_csv('planetary_data/kepler_download_metadata.csv',sep=',',index=False)

    i += 1

kepler_download_df.to_csv('planetary_data/kepler_download_metadata.csv',sep=',',index=False)

logger.info('Task Completed!')


# TODO List:
#
# - Run the models using the graphical input data
# - Run the models using the time series input data
# - Run the models using ensemble model on the input data
# - Complete/close the model building
# - Compare the results with existing models
# - Make PPT summarising the entire work
# - Make Video presentation
# - Submit the work on time by 7th Nov.

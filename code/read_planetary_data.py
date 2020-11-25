import pandas as pd
import os
os.chdir("..")

# colnames="kepid|            tm_designation|  teff| teff_err1| teff_err2|    logg| logg_err1| logg_err2|    feh| feh_err1| feh_err2|     mass| mass_err1| mass_err2|  radius| radius_err1| radius_err2|       dens|  dens_err1|  dens_err2|             prov_sec| kepmag| limbdark_coeff1| limbdark_coeff2| limbdark_coeff3| limbdark_coeff4|      dist| dist_err1| dist_err2| nconfp| nkoi| ntce|                                                           datalink_dvr|              st_delivname| st_vet_date_str|         ra|        dec|          st_quarters|            teff_prov|            logg_prov|             feh_prov|     jmag| jmag_err|     hmag| hmag_err|     kmag| kmag_err| dutycycle|   dataspan|     mesthres01p5|     mesthres02p0|     mesthres02p5|     mesthres03p0|     mesthres03p5|     mesthres04p5|     mesthres05p0|     mesthres06p0|     mesthres07p5|     mesthres09p0|     mesthres10p5|     mesthres12p0|     mesthres12p5|     mesthres15p0|     rrmscdpp01p5|     rrmscdpp02p0|     rrmscdpp02p5|     rrmscdpp03p0|     rrmscdpp03p5|     rrmscdpp04p5|     rrmscdpp05p0|     rrmscdpp06p0|     rrmscdpp07p5|     rrmscdpp09p0|     rrmscdpp10p5|     rrmscdpp12p0|     rrmscdpp12p5|     rrmscdpp15p0|     av| av_err1| av_err2| dutycycle_post| dataspan_post| timeout01p5| timeout02p0| timeout02p5| timeout03p0| timeout03p5| timeout04p5| timeout05p0| timeout06p0| timeout07p5| timeout09p0| timeout10p5| timeout12p0| timeout12p5| timeout15p0| timeoutsumry| cdppslplong| cdppslpshrt"
# colnames = [x.strip(' ') for x in colnames.split("|")]
#
# df = pd.read_table('res/planetary_data/nph-nstedAPI.webarchive',
#                    skiprows=209,names=colnames,nrows=200038,delim_whitespace=True)
# df['tm_designation'] = df['kepid']+' '+df['tm_designation']
# df.drop(['kepid'],inplace=True,axis=1)
# df.reset_index(drop=False,inplace=True)
# df.rename(columns={'index':'kepid'},inplace=True)
# df.to_csv('res/planetary_data/planetary_data_kepler_mission.csv',sep=',',index=False)

df = pd.read_csv('res/planetary_data/planetary_data_kepler_mission.csv',sep=',')
# print(df)
# df = df.loc[df['nkoi']>0]
# print(df.groupby('nconfp').count().reset_index()['kepid'])
# print(df[['mesthres04p5']])

print(df.loc[df['kepid']==10081119])
print(df.sort_values('nkoi',ascending=False)[['kepid','nkoi']])

# 1687 CP
# 8229 KOI
# 200K Total



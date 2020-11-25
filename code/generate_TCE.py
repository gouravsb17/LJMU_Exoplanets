import lightkurve as lk
import matplotlib.pyplot as plt
import os
from astropy.time import Time
os.chdir('..')

def my_custom_corrector_func(lc):
    print(lc.estimate_cdpp())
    # .remove_outliers().remove_nans()
    corrected_lc = lc.remove_nans().normalize().flatten(window_length=401)
    print(corrected_lc.estimate_cdpp())
    return corrected_lc

lc_file = lk.lightcurvefile.KeplerLightCurveFile('simulated_data/kplr000757076-2011271113734_INJECTED-inj1_llc.fits')

corrected_lc = my_custom_corrector_func(lc_file.PDCSAP_FLUX)
corrected_lc.scatter()
plt.show()

exit()

kepler_id = '10525077'
lc_list_files = []
for lc_file in os.listdir('res/kepler_ID_'+kepler_id+'/'):
    if('llc.fits' in lc_file):
        lc_list_files.append(lk.lightcurvefile.KeplerLightCurveFile('res/kepler_ID_'+kepler_id+'/'+lc_file))

lc_collection = lk.LightCurveFileCollection(lc_list_files)
stitched_lc_PDCSAP = lc_collection.PDCSAP_FLUX.stitch()

corrected_lc = my_custom_corrector_func(stitched_lc_PDCSAP)
corrected_lc.scatter()
plt.show()
exit()
# print(corrected_lc.to_pandas())

my_custom_corrector_func(stitched_lc_PDCSAP).fold(period=854.083000).bin().scatter()
plt.show()

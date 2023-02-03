#%%
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import glob

#%%
# Fitting of Gaussian to spectra at room temperature

# Data acquisition
orange_roomT_2V5_190mA_p =  'D://Universidad//Imperial//Year 4//Laboratorio//Characterisation of LEDs and diodes//Data//spectrometer//orange_LED//orange_2V5_190mA_295K.csv'
GaAs_roomT_1V5_187mA_p = 'D://Universidad//Imperial//Year 4//Laboratorio//Characterisation of LEDs and diodes//Data//spectrometer//GaAs_1V5_187mA_294K7.csv'
green_roomT_3V_55mA_p = 'D://Universidad//Imperial//Year 4//Laboratorio//Characterisation of LEDs and diodes//Data//spectrometer//green_3V_55mA_296K4.csv'

orange_roomT_2V5_190mA = pd.read_csv(orange_roomT_2V5_190mA_p, delimiter=';')
orange_roomT_2V5_190mA = orange_roomT_2V5_190mA[orange_roomT_2V5_190mA['Wavelength(nm)']>400]
GaAs_roomT_1V5_187mA = pd.read_csv(GaAs_roomT_1V5_187mA_p, delimiter=';')
GaAs_roomT_1V5_187mA = GaAs_roomT_1V5_187mA[GaAs_roomT_1V5_187mA['Wavelength(nm)']>400]
green_roomT_3V_55mA = pd.read_csv(green_roomT_3V_55mA_p)
green_roomT_3V_55mA = green_roomT_3V_55mA[green_roomT_3V_55mA['Wavelength(nm)']>400]

# Plotting
plt.figure(figsize=(15, 10))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Amplitude (au)')

plt.plot(orange_roomT_2V5_190mA.iloc[:, 0], orange_roomT_2V5_190mA.iloc[:, 1], color='orange', label=str(2.5*0.190)+' W')
plt.plot(GaAs_roomT_1V5_187mA.iloc[:, 0], GaAs_roomT_1V5_187mA.iloc[:, 1], color='brown', label=str(np.format_float_positional(1.5*0.187, precision=3))+' W')
plt.plot(green_roomT_3V_55mA.iloc[:, 0], green_roomT_3V_55mA.iloc[:, 1], color='green', label=str(np.format_float_positional(3.55*0.055, precision=3))+' W')



# Fitting
def gaussian(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

fit_green = green_roomT_3V_55mA[green_roomT_3V_55mA['Wavelength(nm)']>500]
fit_green = fit_green[fit_green['Wavelength(nm)']<680]

fit_orange = orange_roomT_2V5_190mA[orange_roomT_2V5_190mA['Wavelength(nm)']>540]
fit_orange = fit_orange[fit_orange['Wavelength(nm)']<700]

fit_GaAs = GaAs_roomT_1V5_187mA[GaAs_roomT_1V5_187mA['Wavelength(nm)']>950]


vals_green, cov_green = opt.curve_fit(gaussian, fit_green.iloc[:, 0], fit_green.iloc[:, 1], p0=(0.6, np.mean(fit_green.iloc[:, 0]), 50))
vals_orange, cov_orange = opt.curve_fit(gaussian, fit_orange.iloc[:, 0], fit_orange.iloc[:, 1], p0=(0.6, np.mean(fit_orange.iloc[:, 0]), 50))
vals_GaAs, cov_GaAs = opt.curve_fit(gaussian, fit_GaAs.iloc[:, 0], fit_GaAs.iloc[:, 1], p0=(0.6, np.mean(fit_GaAs.iloc[:, 0]), 50))


plt.plot(green_roomT_3V_55mA.iloc[:, 0], gaussian(green_roomT_3V_55mA.iloc[:, 0], *vals_green), color="#32cd32")
plt.plot(orange_roomT_2V5_190mA.iloc[:, 0], gaussian(orange_roomT_2V5_190mA.iloc[:, 0], *vals_orange), color="#FF8C00")
plt.plot(GaAs_roomT_1V5_187mA.iloc[:, 0], gaussian(GaAs_roomT_1V5_187mA.iloc[:, 0], *vals_GaAs), color="#5C4033")


# Find second peak of GaAs
second_peak_data = GaAs_roomT_1V5_187mA.iloc[:, 1] - gaussian(GaAs_roomT_1V5_187mA.iloc[:, 0], *vals_GaAs)
second_peak_data = second_peak_data[:2439]
second_peak_data = pd.DataFrame([GaAs_roomT_1V5_187mA.iloc[:, 0].to_numpy()[:len(second_peak_data)], second_peak_data]).T
plt.plot(second_peak_data.iloc[:, 0], second_peak_data.iloc[:, 1], color="pink", label="Second peak data")

second_fit = second_peak_data[second_peak_data.iloc[:, 0]>820]
vals_2_GaAs, cov_2_GaAs = opt.curve_fit(gaussian, second_fit.iloc[:, 0], second_fit.iloc[:, 1], p0=(0.15, np.mean(second_fit.iloc[:, 0]), 50))
plt.plot(GaAs_roomT_1V5_187mA.iloc[:, 0], gaussian(GaAs_roomT_1V5_187mA.iloc[:, 0], *vals_2_GaAs), color="red", label="2nd peak")

print(vals_2_GaAs)

plt.plot(GaAs_roomT_1V5_187mA.iloc[:, 0], gaussian(GaAs_roomT_1V5_187mA.iloc[:, 0], *vals_2_GaAs)+gaussian(GaAs_roomT_1V5_187mA.iloc[:, 0], *vals_GaAs), color="black", label="Combined")

plt.vlines(vals_green[1], 0, 0.72, label=str(vals_green[1]), color="")
plt.vlines(vals_orange[1], 0, 0.72, label=str(vals_orange[1]), color="")
plt.vlines(vals_2_GaAs[1], 0, 0.72, label=str(vals_2_GaAs[1]), color="")
plt.vlines(vals_GaAs[1], 0, 0.72, label=str(vals_GaAs[1]), color="")


plt.grid()
plt.legend()
plt.show()




#%%
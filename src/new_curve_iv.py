import ROOT
import matplotlib.pyplot as plt
from utilities import *
from uncertainties import umath
import pandas as pd
import numpy as np
from scipy import constants as const
from scipy.optimize import curve_fit
import scipy.constants
import scipy.odr as odr


DATADIR = "../data"
OUTDIR = "../output"

#read dataframes iv-curves

iv_48 = "iv_48_0_2e-3_100pt"
I_48, V_48, Verr_48 = iv_dataframe_to_array(iv_48)


iv_51 = "iv_51_5_1-5e-3_100pt"
I_51, V_51, Verr_51 = iv_dataframe_to_array(iv_51)


iv_53 = "iv_53_5_1e-3_100pt"
I_53, V_53, Verr_53 = iv_dataframe_to_array(iv_53)


iv_54 = "iv_54_5_1e-3_100pt"
I_54, V_54, Verr_54 = iv_dataframe_to_array(iv_54)


iv_55 = "iv_55_5_1e-3_100pt"
I_55, V_55, Verr_55 = iv_dataframe_to_array(iv_55)


iv_56 = "iv_56_5_1e-3_100pt"
I_56, V_56, Verr_56 = iv_dataframe_to_array(iv_56)


iv_57 = "iv_57_5_1e-3_100pt"
I_57, V_57, Verr_57 = iv_dataframe_to_array(iv_57)


iv_62 = "iv_62_0_1e-3_100pt"
I_62, V_62, Verr_62 = iv_dataframe_to_array(iv_62)


         ######################
         ##                  ## 
         ##      PLOTS       ##
         ##                  ##
         ######################

title_size = 15
label_size = 12

####### ZOOM #############

fig2, ax2 = plt.subplots(figsize=(8,6))

iv_53_zoom = "iv_53_5_1e-3_100pt_zoom"
I_53_zoom, V_53_zoom, Verr8 = iv_dataframe_to_array(iv_53_zoom)

ax2.errorbar(I_53_zoom, V_53_zoom, marker="o", linewidth = 0, markersize=3, label="T = 53.5 K - zoom")
ax2.set_xlabel('I[mA]', fontsize = label_size)
ax2.set_ylabel('V[mV]', fontsize = label_size)
ax2.set_title("I-V curve at 53.5 K ", fontsize = title_size)
ax2.legend(loc="lower right")
ax2.grid()
fig2.savefig(f"{OUTDIR}/zoom_curve.png")

## Determining the error on V

#x = np.linspace(0,100,1)
zoom_branch_1 = V_53_zoom[1:100]
zoom_branch_2 = V_53_zoom[101:200]
zoom_branch_2 = zoom_branch_2[::-1]

zoom_branch_3 = V_53_zoom[201:300]
zoom_branch_4 = V_53_zoom[301:400]
zoom_branch_4 = zoom_branch_2[::-1]

deltaV_distribution1 = abs(zoom_branch_1-zoom_branch_2)
#deltaV_distribution2 = abs(zoom_branch_3-zoom_branch_4)
#deltaV_distribution = np.concatenate((deltaV_distribution1,deltaV_distribution2))

V_err = (max(deltaV_distribution1)-min(deltaV_distribution1))/2 # mV
#V_err = (max(deltaV_distribution1)) # mV
print(V_err)


####### ALL CURVES ##########

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.errorbar(I_48,V_48, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 48.0 K", elinewidth = 0.5, ecolor = 'k', mfc = 'cyan', mew = 0)
ax1.errorbar(I_51,V_51, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 51.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'pink', mew = 0)
ax1.errorbar(I_53,V_53, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 53.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'blue', mew = 0)
ax1.errorbar(I_54,V_54, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 54.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'orange', mew = 0)
ax1.errorbar(I_55,V_55, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 55.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'green', mew = 0)
ax1.errorbar(I_56,V_56, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 56.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'red', mew = 0)
ax1.errorbar(I_57,V_57, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 57.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'purple', mew = 0)
ax1.errorbar(I_62,V_62, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 62.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'olive', mew = 0)

ax1.set_xlabel('I[mA]', fontsize = label_size)
ax1.set_ylabel('V[mV]', fontsize = label_size)
ax1.set_title("I-V curves at different temperatures", fontsize = title_size)
ax1.legend(loc="lower right")
ax1.grid()
fig1.savefig(f"{OUTDIR}/all_curves.png")


### List of parameters ###

T = [48.5, 51.5, 53.5, 54.5, 55.5, 56.5]
I_th = []
I_th_err = []
#G = scipy.constants.k*T/I_th
#G_err = []



### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")

def func_fit(x, y0, A, I_th):
    return y0 + A*np.sinh(x/I_th)

####################
#### FIT 48.5 ######
####################


print("----------------------------")
print("|                          |")
print("|      Fit T = 48.5 K      |")
print("|                          |")
print("----------------------------")

ndata_48 = len(I_48)
Verr_48 = 10*np.ones(ndata_48)*V_err
Ierr_48 = np.zeros(ndata_48)

######## scipy optimize ######


param48, cov48 = curve_fit(func_fit, I_48, V_48, sigma = Verr_48)
print(param48)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_48, V_48, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_48, func_fit(I_48, *param48), 'r-', label='fit: y0=%5.3f, A=%5.3f, I_th=%5.3f' % tuple(param48))

ax3.set_xlabel('I[mA]', fontsize = label_size)
ax3.set_ylabel('V[mV]', fontsize = label_size)
ax3.set_title("I-V at 48.5 K", fontsize = title_size)
fig3.savefig(f"{OUTDIR}/fit_48_5.png")
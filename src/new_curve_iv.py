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

ax1.set_xlabel('I [mA]', fontsize = label_size)
ax1.set_ylabel('V [mV]', fontsize = label_size)
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


def func_fit(x, y0, A, I_th):
    return y0 + A*np.sinh(x/I_th)

param_label = ["y_0", "A", "I_th"]
param_units = ["mV", "mV", "mA"]

####################
#### FIT 48.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


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
#print(param48)
param_err48 = np.sqrt(np.diag(cov48))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_48, V_48, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_48, func_fit(I_48, *param48), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param48[0], param_err48[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param48[1], param_err48[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param48[2], param_err48[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 48.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_48_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param48[i], " +- ", param_err48[i], " ", param_units[i], "\n")

I_th.append(param48[2])
I_th_err.append(param_err48[2])


####################
#### FIT 51.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 51.5 K      |")
print("|                          |")
print("----------------------------")

ndata_51 = len(I_51)
Verr_51 = 10*np.ones(ndata_51)*V_err
Ierr_51 = np.zeros(ndata_51)

######## scipy optimize ######

param51, cov51 = curve_fit(func_fit, I_51, V_51, sigma = Verr_51)
#print(param48)
param_err51 = np.sqrt(np.diag(cov51))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_51, V_51, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_51, func_fit(I_51, *param51), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param51[0], param_err51[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param51[1], param_err51[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param51[2], param_err51[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 51.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_51_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param51[i], " +- ", param_err51[i], " ", param_units[i], "\n")

I_th.append(param51[2])
I_th_err.append(param_err51[2])


####################
#### FIT 53.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 53.5 K      |")
print("|                          |")
print("----------------------------")

ndata_53 = len(I_53)
Verr_53 = 10*np.ones(ndata_53)*V_err
Ierr_53 = np.zeros(ndata_53)

######## scipy optimize ######

param53, cov53 = curve_fit(func_fit, I_53, V_53, sigma = Verr_53)
#print(param48)
param_err53 = np.sqrt(np.diag(cov53))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_53, V_53, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_53, func_fit(I_53, *param53), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param53[0], param_err53[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param53[1], param_err53[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param53[2], param_err53[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 53.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_53_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param53[i], " +- ", param_err53[i], " ", param_units[i], "\n")

I_th.append(param53[2])
I_th_err.append(param_err53[2])


####################
#### FIT 54.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 54.5 K      |")
print("|                          |")
print("----------------------------")

ndata_54 = len(I_54)
Verr_54 = 10*np.ones(ndata_54)*V_err
Ierr_54 = np.zeros(ndata_54)

######## scipy optimize ######

param54, cov54 = curve_fit(func_fit, I_54, V_54, sigma = Verr_54)
#print(param48)
param_err54 = np.sqrt(np.diag(cov54))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_54, V_54, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_54, func_fit(I_54, *param54), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param54[0], param_err54[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param54[1], param_err54[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param54[2], param_err54[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 54.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_54_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param54[i], " +- ", param_err54[i], " ", param_units[i], "\n")

I_th.append(param54[2])
I_th_err.append(param_err54[2])




print(I_th)


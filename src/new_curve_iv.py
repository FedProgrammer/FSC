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

T = np.array([48.5, 51.5, 53.5, 54.5, 55.5, 56.5, 57.5, 62.0]) ## Kelvin
I_th = []
I_th_err = []
G = []
G_err = []



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

####################
#### FIT 55.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 55.5 K      |")
print("|                          |")
print("----------------------------")

ndata_55 = len(I_55)
Verr_55 = 10*np.ones(ndata_55)*V_err
Ierr_55 = np.zeros(ndata_55)

######## scipy optimize ######

param55, cov55 = curve_fit(func_fit, I_55, V_55, sigma = Verr_55)
#print(param48)
param_err55 = np.sqrt(np.diag(cov55))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_55, V_55, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_55, func_fit(I_55, *param55), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param55[0], param_err55[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param55[1], param_err55[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param55[2], param_err55[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 55.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_55_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param55[i], " +- ", param_err55[i], " ", param_units[i], "\n")

I_th.append(param55[2])
I_th_err.append(param_err55[2])


####################
#### FIT 56.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 56.5 K      |")
print("|                          |")
print("----------------------------")

ndata_56 = len(I_56)
Verr_56 = 10*np.ones(ndata_56)*V_err
Ierr_56 = np.zeros(ndata_56)

######## scipy optimize ######

param56, cov56 = curve_fit(func_fit, I_56, V_56, sigma = Verr_56, bounds=([-10e-2,0,0],[10, 1, 1]))
#print(param48)
param_err56 = np.sqrt(np.diag(cov56))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_56, V_56, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_56, func_fit(I_56, *param56), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param56[0], param_err56[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param56[1], param_err56[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param56[2], param_err56[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 56.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_56_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param56[i], " +- ", param_err56[i], " ", param_units[i], "\n")

I_th.append(param56[2])
I_th_err.append(param_err56[2])


####################
#### FIT 57.5 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 57.5 K      |")
print("|                          |")
print("----------------------------")

ndata_57 = len(I_57)
Verr_57 = 10*np.ones(ndata_57)*V_err
Ierr_57 = np.zeros(ndata_57)

######## scipy optimize ######

param57, cov57 = curve_fit(func_fit, I_57, V_57, sigma = Verr_57, bounds=([-10e-2,0,0],[10, 1, 1]))
#print(param48)
param_err57 = np.sqrt(np.diag(cov57))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_57, V_57, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_57, func_fit(I_57, *param57), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param57[0], param_err57[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param57[1], param_err57[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param57[2], param_err57[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 57.5 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_57_5.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param57[i], " +- ", param_err57[i], " ", param_units[i], "\n")

I_th.append(param57[2])
I_th_err.append(param_err57[2])




####################
#### FIT 62.0 ######
####################

### Fitting function according Kim-Anderson Model ###
print("\n\nFitting function according Kim-Anderson model: \n")
print("\t\t V = y_0 + A*sinh(I/I_th)\n\n")


print("----------------------------")
print("|                          |")
print("|      Fit T = 62.0 K      |")
print("|                          |")
print("----------------------------")

ndata_62 = len(I_62)
Verr_62 = 10*np.ones(ndata_62)*V_err
Ierr_62 = np.zeros(ndata_62)

######## scipy optimize ######

param62, cov62 = curve_fit(func_fit, I_62, V_62, sigma = Verr_62)
#print(param48)
param_err62 = np.sqrt(np.diag(cov62))

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_62, V_62, marker="o", markersize=3, color = "blue", label='data', linewidth = 0)
ax3.plot(I_62, func_fit(I_62, *param62), 'r-', label='fitted curve')

textstr = '\n'.join((
    r'$\bf{Fit \ parameters: }$',
    r'$y_0 = %.5f \pm %.6f $ mV' % (param62[0], param_err62[0]),
    r'$A = %.7f \pm %.7f$ mV' % (param62[1], param_err62[1]),
    r'$I_{th} = %.3f \pm %.3f$ mA' % (param62[2], param_err62[2])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax3.grid()
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel('V [mV]', fontsize = label_size)
ax3.set_title("I-V at T = 62.0 K", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/fit_62_0.png")

print("\n\nFitting results:\n")
for i in range(3):
    print(param_label[i]," = ", param62[i], " +- ", param_err62[i], " ", param_units[i], "\n")

I_th.append(param62[2])
I_th_err.append(param_err62[2])


####################
#### I_th vs T #####
####################

k = scipy.constants.Boltzmann
I_th = np.array(I_th)
I_th_err = np.array(I_th_err)

G = k*T/(I_th*0.001) # J/A
G_err = k*T*I_th_err*0.001 / pow(I_th*0.001,2) #J/A

print("\n\n I_th vs T\n")
for i in range(8):
    print(T[i]," K \t I_th = ", I_th[i], " +- ", I_th_err[i], " mA \t G = ", G[i], " +- ", G_err[i]," J/A \n")


fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.errorbar(T, I_th,yerr=I_th_err, marker="o", markersize=8, color = "blue", label='data', linewidth = 0, elinewidth = 1.5, ecolor = 'k', mfc = 'purple', mew = 0)
ax3.grid()
ax3.set_xlabel('T [K]', fontsize = label_size)
ax3.set_ylabel(r'$I_{th} $ [mA]', fontsize = label_size)
ax3.set_title(r"$I_{th} \ vs \ T$", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/i_th_vs_Temp.png")

####################
####   G vs T  #####
####################

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.errorbar(T, G,yerr=G_err, marker="o", markersize=8, color = "blue", label='data', linewidth = 0, elinewidth = 1.5, ecolor = 'k', mfc = 'green', mew = 0)
ax3.grid()
ax3.set_xlabel('T [K]', fontsize = label_size)
ax3.set_ylabel(r'$G $ [J/A]', fontsize = label_size)
ax3.set_title(r"$G \ vs \ T$", fontsize = title_size)
ax3.legend(loc="lower right")
fig3.savefig(f"{OUTDIR}/G_vs_Temp.png")
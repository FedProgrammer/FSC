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
from scipy.signal import find_peaks

DATADIR = "../data/flux"
OUTDIR = "../output"


title_size = 15
label_size = 12
V_offset = 4.614 ## uV


def sinusoidal_fit(x, y_0, A, x_c, w):
    return y_0+A*np.sin(np.pi*(x-x_c)/w)

### squid V-Phi at 77K V_max1

flux_max_1 = "06_01_2023-15_59_36_.csv"
i_1, v_1 = squid_dataframe_to_array2(flux_max_1)

### squid V-Phi at 77K V_max2

flux_max_2 = "06_01_2023-16_00_21_.csv"
i_2, v_2 = squid_dataframe_to_array2(flux_max_2)

### squid V-Phi at 77K V_max3

flux_max_3 = "06_01_2023-16_09_00_.csv"
i_3, v_3 = squid_dataframe_to_array2(flux_max_3)


'''
##############################
####  OFFSET TRANSLATION  ####
##############################

V_77_min += V_offset
V_77_max += V_offset


## selecting data for linear fit ##

i_lin1 = I_77_max[I_77_max > 80 ]
v_lin1 = V_77_max[I_77_max > 80 ]

i_lin = i_lin1[i_lin1 < 110 ]
v_lin = v_lin1[i_lin1 < 110 ]

param_Ic_max, cov_Ic_max = curve_fit(linear_fit, i_lin, v_lin)
param_err_Ic_max = np.sqrt(np.diag(cov_Ic_max))


x_line2=np.linspace(0,80,1000)

#param_77_min, cov_77_min = curve_fit(linear_fit, I_77_min, V_77_min)
#param_err_77_min = np.sqrt(np.diag(cov_77_min))
'''

###############
#### MAX 1 ####
###############


param_i_1, cov_i_1 = curve_fit(sinusoidal_fit, i_1, v_1, bounds=([4,4,10,40],[6,5,20,60]))
param_err_i_1 = np.sqrt(np.diag(cov_i_1))
x_line=np.linspace(min(i_1),max(i_1),1000)

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(i_1, v_1, linewidth = 1, linestyle = '-', label=r"V - internal flux")
#ax1.plot(I_77_min, V_77_min, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
ax1.plot(x_line, sinusoidal_fit(x_line,*param_i_1), linewidth = 2,linestyle="dashed", label=r"sinusoidal fit: voltage modulation", color = 'green', zorder=2)
#ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"V-$\Phi$ characteristic at 77 K - attempt #1", fontsize = title_size)
#ax1.set_xlim(0,max(I_77_max))
#ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc="lower left")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Sinusoidal \ fit \ parameters: }$',
    r'y_0 = %.2f$ \pm $ %.2f $\mu$V' % (param_i_1[0], param_err_i_1[0]),
    r'A = %.2f$ \pm $ %.2f $\mu$V' % (param_i_1[1], param_err_i_1[1]),
    r'$I_c$ = %.2f$ \pm $ %.2f $\mu$A' % (param_i_1[2], param_err_i_1[2]),
    r'$w$ = %.2f$ \pm $ %.2f  $\mu$A' % (param_i_1[3], param_err_i_1[3])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.9)

ax1.text(0.60, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_v_flux_1.png")



print("----------------------------")
print("|       Squid @77          |")
print("|     V modulation 1       |")
print("----------------------------")

V_mod1 = 2*param_i_1[1]
V_mod_err1 = 2*param_err_i_1[1]
I_period1 = 2*param_i_1[3]
I_period_err1 = 2*param_err_i_1[3]

print("\n V modulation 1 = %f +- %f Ohm\n" %(V_mod1, V_mod_err1))
print("\n I period = %f +- %f  uA\n" %(I_period1, I_period_err1))



###############
#### MAX 2 ####
###############


param_i_2, cov_i_2 = curve_fit(sinusoidal_fit, i_2, v_2, bounds=([4,4,10,40],[6,5,20,60]))
param_err_i_2 = np.sqrt(np.diag(cov_i_2))
x_line=np.linspace(min(i_2),max(i_2),1000)

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(i_2, v_2, linewidth = 1, linestyle = '-', label=r"V - internal flux")
#ax1.plot(I_77_min, V_77_min, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
ax1.plot(x_line, sinusoidal_fit(x_line,*param_i_2), linewidth = 2,linestyle="dashed", label=r"sinusoidal fit: voltage modulation", color = 'green', zorder=2)
#ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"V-$\Phi$ characteristic at 77 K - attempt #2", fontsize = title_size)
#ax1.set_xlim(0,max(I_77_max))
#ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc="lower left")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Sinusoidal \ fit \ parameters: }$',
    r'y_0 = %.2f$ \pm$ %.2f $\mu$V' % (param_i_2[0], param_err_i_2[0]),
    r'A = %.2f$ \pm $ %.2f $\mu$V' % (param_i_2[1], param_err_i_2[1]),
    r'$I_c$ = %.2f$ \pm $ %.2f $\mu$A' % (param_i_2[2], param_err_i_2[2]),
    r'$w$ = %.2f$ \pm  $ %.2f $\mu$A' % (param_i_2[3], param_err_i_2[3])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.9)

ax1.text(0.60, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_v_flux_2.png")



print("----------------------------")
print("|       Squid @77          |")
print("|     V modulation 2       |")
print("----------------------------")

V_mod2 = 2*param_i_2[1]
V_mod_err2 = 2*param_err_i_2[1]
I_period2 = 2*param_i_2[3]
I_period_err2 = 2*param_err_i_2[3]

print("\n V modulation 2 = %f +- %f Ohm\n" %(V_mod2, V_mod_err2))
print("\n I period = %f +- %f  uA\n" %(I_period2, I_period_err2))


###############
#### MAX 3 ####
###############

param_i_3, cov_i_3 = curve_fit(sinusoidal_fit, i_3, v_3, bounds=([4,4,10,40],[6,5,20,60]))
param_err_i_3 = np.sqrt(np.diag(cov_i_3))
x_line=np.linspace(min(i_3),max(i_3),1000)

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(i_3, v_3, linewidth = 1, linestyle = '-', label=r"V - internal flux")
#ax1.plot(I_77_min, V_77_min, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
ax1.plot(x_line, sinusoidal_fit(x_line,*param_i_3), linewidth = 2,linestyle="dashed", label=r"sinusoidal fit: voltage modulation", color = 'green', zorder=2)
#ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"V-$\Phi$ characteristic at 77 K - attempt #3", fontsize = title_size)
#ax1.set_xlim(0,max(I_77_max))
#ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc="lower left")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Sinusoidal \ fit \ parameters: }$',
    r'y_0 = %.2f$ \pm$ %.2f $\mu$V' % (param_i_3[0], param_err_i_3[0]),
    r'A = %.2f$ \pm $ %.2f $\mu$V' % (param_i_3[1], param_err_i_3[1]),
    r'$I_c$ = %.2f$ \pm $ %.2f $\mu$A' % (param_i_3[2], param_err_i_3[2]),
    r'$w$ = %.2f$ \pm $ %.2f $\mu$A' % (param_i_3[3], param_err_i_3[3])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.9)

ax1.text(0.60, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_v_flux_3.png")



print("----------------------------")
print("|       Squid @77          |")
print("|     V modulation 3       |")
print("----------------------------")

V_mod3 = 2*param_i_3[1]
V_mod_err3 = 2*param_err_i_3[1]
I_period3 = 2*param_i_3[3]
I_period_err3 = 2*param_err_i_3[3]

print("\n V modulation 3 = %f +- %f Ohm\n" %(V_mod3, V_mod_err3))
print("\n I period = %f +- %f  uA\n" %(I_period3, I_period_err3))

I_period_mean = np.mean([I_period1, I_period2, I_period3])
I_period_std = (max([I_period1, I_period2, I_period3])-min([I_period1, I_period2, I_period3]))*0.5

print("\n I period mean = %f +- %f  uA\n" %(I_period_mean, I_period_std))

phi_0 = scipy.constants.Planck/(2*scipy.constants.e)

#Mutual inductuance

M = phi_0 / (I_period_mean*1e-6) # H
M_err = phi_0 * I_period_std*(1e-6) / pow((I_period_mean*1e-6),2) # H

print("\n Mutual inductance M = %f +- %f  pH\n" %(M*1e12, M_err*1e12))



###############
### V-Phi 1 ###
###############

flux = M*i_1*1e-6 ## Wb

param_flux, cov_flux = curve_fit(sinusoidal_fit, flux, v_1,  bounds=([4,4,12*M*1e-6,48*M*1e-6],[6,5,13*M*1e-6,50*M*1e-6]))
param_err_flux = np.sqrt(np.diag(cov_flux))
x_line=np.linspace(min(flux),max(flux),1000)

#derivative

v_1_prime = np.diff(sinusoidal_fit(x_line,*param_flux)) / np.diff(x_line)
x_prime = (np.array(x_line)[:-1] + np.array(x_line)[1:]) / 2

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(flux, v_1, linewidth = 1, linestyle = '-', label=r"real V - $\Phi$ char.")
#ax1.plot(x_prime, v_1_prime, linewidth = 1, linestyle = '-', label=r"$\dv{V}{\Phi}")
ax1.plot(x_line, sinusoidal_fit(x_line,*param_flux), linewidth = 2,linestyle="dashed", label=r"sinusoidal fit", color = 'green', zorder=2)
#ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'$\Phi$ [Wb]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"V-$\Phi$ characteristic at 77 K", fontsize = title_size)
#ax1.set_xlim(0,max(I_77_max))
#ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc="lower left")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Sinusoidal \ fit \ parameters: }$',
    r'y_0 = %.2f$ \pm $ %.2f $\mu$V' % (param_flux[0], param_err_flux[0]),
    r'A = %.2f$ \pm $ %.2f $\mu$V' % (param_flux[1], param_err_flux[1]),
    r'$\Phi_c$ = %.2f$ \pm $ %.2f fWb' % (param_flux[2]*1e15, param_err_flux[2]*1e15),
    r'$w_{Phi}$ = %.2f$ \pm $ %.2f  fWb' % (param_flux[3]*1e15, param_err_flux[3]*1e15)))

props = dict(boxstyle='round', facecolor='grey', alpha=0.9)

ax1.text(0.60, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_v_flux_peaks.png")



print("----------------------------")
print("|       Squid @77          |")
print("|       V - flux           |")
print("----------------------------")

print("trasfer function = ", max(v_1_prime), " V / Wb" )




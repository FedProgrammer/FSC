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
from scipy.signal import savgol_filter, find_peaks


DATADIR = "../data"
OUTDIR = "../output"

rt_char_file = "caratterizzazione_R_T"
time, T_diode, T_sample, V, V_err, R = rt_dataframe_to_array(rt_char_file)

I = 3 ## uA
R_err = V_err / I # Ohm

         ######################
         ##                  ## 
         ##      PLOTS       ##
         ##                  ##
         ######################

title_size = 15
label_size = 12

fig2, ax2 = plt.subplots(figsize=(8,6))

R_prime = np.diff(R) / np.diff(T_sample)
T_prime = (np.array(T_sample)[:-1] + np.array(T_sample)[1:]) / 2
R_prime_smooth = savgol_filter(R_prime, 15, 2) 
R_prime_smooth2 = savgol_filter(R_prime, 25, 3) 


ax2.errorbar(T_sample, R, yerr = R_err, marker="o", linewidth = 0, markersize=3, label="RT", elinewidth = 0.5, ecolor = 'k', mfc = 'blue', mew = 0)
ax2.plot(T_prime, R_prime, linewidth = 1, linestyle = '-', label="1st der", color = 'red', alpha = 0.5)
ax2.plot(T_prime, R_prime_smooth[:708], linewidth = 2, linestyle = '-', label="1st der - smoothing", color = 'green')
ax2.plot(T_prime, R_prime_smooth2[:708], linewidth = 2, linestyle = '-', label="1st der - smoothing", color = 'pink')
ax2.set_xlabel(r'$T_{sample}$ [K]', fontsize = label_size)
ax2.set_ylabel(r'$Resistance$ [$\mu$V]', fontsize = label_size)
ax2.set_title(r"R-T characteristic at I = 3$\mu$A ", fontsize = title_size)
ax2.legend(loc="upper left")
ax2.grid()
fig2.savefig(f"{OUTDIR}/RT_char.png")


## ZOOM ##

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(T_prime, R_prime, linewidth = 1, linestyle = '-', label="1st der", color = 'red', alpha = 0.5)
ax3.plot(T_prime, R_prime_smooth[:708], linewidth = 2, linestyle = '-', label="1st der - smoothing 15pt", color = 'green', alpha = 0.5)
ax3.plot(T_prime, R_prime_smooth2[:708], linewidth = 2, linestyle = '-', label="1st der - smoothing 25pt", color = 'purple')
ax3.set_xlabel(r'$T_{sample}$ [K]', fontsize = label_size)
ax3.set_ylabel(r'$Resistance$ [$\mu$V]', fontsize = label_size)
ax3.set_title(r"R-T characteristic at I = 3$\mu$A ", fontsize = title_size)
ax3.legend(loc="upper left")
ax3.set_xlim(60,80)
ax3.grid()
fig3.savefig(f"{OUTDIR}/RT_char_zoom_.png")

peaks, _ = find_peaks(R_prime_smooth2, height = 17)
print(T_prime[peaks])
Tmid = np.mean(T_prime[peaks])
Tmid_err = np.std(T_prime[peaks])
print(Tmid)
print(Tmid_err)
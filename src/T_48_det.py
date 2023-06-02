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
ndata_48 = len(I_48)
#Verr_48 = 10*np.ones(ndata_48)*V_err
Ierr_48 = np.zeros(ndata_48)

#select the positive branch

I_pos = I_48[I_48>0]
V_pos = V_48[I_48>0]

# add an offset to have positive value for logarithm
V_pos = V_pos + 0.003

# noise dashed line
x_noise = np.linspace(0,1.75,500)
y_noise = 6.5*1e-4*np.ones(500)

#select linear values
I_lin = I_pos[I_pos>1]
V_lin = V_pos[I_pos>1]

I_lin2 = I_lin[I_lin<1.75]
V_lin2 = V_lin[I_lin<1.75]



####### ZOOM #############
title_size = 15
label_size = 12

fig2, ax2 = plt.subplots(figsize=(8,6))

ax2.errorbar(I_pos, V_pos,    marker="o", linewidth = 0, markersize=3, label="T = 53.5 K - zoom")
ax2.plot(x_noise, y_noise, linewidth = 2,linestyle="dashed", label="noise level", color = 'red', zorder=1)
#ax2.plot(x_line, linear_fit(x_line,*param), linewidth = 2,linestyle="dashed", label="expected exponentiale", color = 'green', zorder=1)
ax2.set_xlabel('I [mA]', fontsize = label_size)
ax2.set_ylabel('V [mV]', fontsize = label_size)
ax2.set_title("I-V curve at 48.0 K - positive branch - lin-log", fontsize = title_size)
ax2.legend(loc="lower right")
ax2.set_yscale("log")
ax2.grid()
fig2.savefig(f"{OUTDIR}/T_48_logscale.png")

####################

def linear_fit(x, m, q):
    return m*x + q

V_log = np.log10(V_lin2[I_lin2>1])
param, cov = curve_fit(linear_fit, I_lin2, V_log)
print(param)

x_line = np.linspace(0.75,2,500)

title_size = 15
label_size = 12

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.errorbar(I_pos, np.log10(V_pos),  marker="o", linewidth = 0, markersize=3, label="T = 53.5 K - zoom", alpha=0.6)
ax3.plot(x_noise, y_noise-3.175, linewidth = 3,linestyle="dashed", label="noise level", color = 'red', zorder=1)
ax3.plot(x_line, linear_fit(x_line,*param), linewidth = 3,linestyle="dashed", label="possible exponential behaviour", color = 'green', zorder=2)
ax3.set_xlabel('I [mA]', fontsize = label_size)
ax3.set_ylabel(r'$\log{V}$', fontsize = label_size)
ax3.set_title("I-log(V) curve at 48.0 K ", fontsize = title_size)
ax3.legend(loc="lower right")
ax3.grid()
fig3.savefig(f"{OUTDIR}/T_48_logy.png")
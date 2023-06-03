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


DATADIR = "../data/iv_RT"
OUTDIR = "../output"


title_size = 15
label_size = 12

def linear_fit(x, m, q):
    return m*x + q

### squid IV char at room temperature

iv_RT = "06_01_2023-15_13_47_.csv"
I_rt, V_rt = squid_dataframe_to_array(iv_RT)

param_RT, cov_RT = curve_fit(linear_fit, I_rt, V_rt)
param_err_RT = np.sqrt(np.diag(cov_RT))

x_line=np.linspace(min(I_rt),max(I_rt),1000)


fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(I_rt, V_rt, linewidth = 1, linestyle = '-', label="I-V char at RT")
ax1.plot(x_line, linear_fit(x_line,*param_RT), linewidth = 1,linestyle="solid", label="linear fit: Ohm's law", color = 'green', zorder=2)
ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"I-V characteristic at room temperature", fontsize = title_size)
ax1.legend(loc="lower right")
ax1.grid()
textstr = '\n'.join((
    r'$\bf{Linear \ fit \ parameters: }$',
    r'R = %.2f$ \pm %.2f \ \Omega$' % (param_RT[0], param_err_RT[0]),
    r'offset = %.2f $\pm %.2f \ \mu$V' % (param_RT[1], param_err_RT[1])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

fig1.savefig(f"{OUTDIR}/squid_iv_rt.png")

print("----------------------------")
print("|       Squid @RT          |")
print("|      linear fit IV       |")
print("----------------------------")

print("\n Slope = %f +- %f Ohm\n" %(param_RT[0], param_err_RT[0]))
print("\n Intercept = %f +- %f  uV\n" %(param_RT[1], param_err_RT[1]))





### squid IV char at 77 K

iv_77 = "06_01_2023-15_31_50_.csv"
I_77, V_77 = squid_dataframe_to_array(iv_77)

I_plat = I_77[I_77 > -50]
V_plat = V_77[I_77 > -50]
I_plat2 = I_plat[I_plat < 50]
V_plat2 = V_plat[I_plat < 50]

param_plat, cov_plat = curve_fit(linear_fit, I_plat2, V_plat2)
param_err_plat = np.sqrt(np.diag(cov_plat))

x_line=np.linspace(-100,100,1000)


fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(I_77, V_77, linewidth = 1, linestyle = '-', label="I-V char at 77 K", alpha = 0.5)
ax1.plot(I_77, V_77-param_plat[1], linewidth = 1, linestyle = '-', label="I-V char at 77 K with offset", color='red', alpha = 1)
ax1.plot(x_line, linear_fit(x_line,*param_plat), linewidth = 1,linestyle="dashed", label="voltage offset", color = 'green', zorder=2)
ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"I-V characteristic at T = 77 K", fontsize = title_size)
ax1.legend(loc="lower right")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Linear \ fit \ parameters: }$',
    r'R = %.2f$ \pm %.2f \ \Omega$' % (param_plat[0], param_err_plat[0]),
    r'offset = %.2f $\pm %.2f \ \mu$V' % (param_plat[1], param_err_plat[1])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_iv_77.png")

print("----------------------------")
print("|       Squid @ 77K        |")
print("|   linear fit plateau     |")
print("----------------------------")

print("\n Slope = %f +- %f Ohm\n" %(param_plat[0], param_err_plat[0]))
print("\n V_offset = %f +- %f  uV\n" %(param_plat[1], param_err_plat[1]))


V_offset = -4.614 ## uV
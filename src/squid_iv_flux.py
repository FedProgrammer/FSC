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
V_offset = 4.614 ## uV


def linear_fit(x, m, q):
    return m*x + q

### squid IV at 77K I_c max

iv_77_max = "06_01_2023-15_41_45_.csv"
I_77_max, V_77_max = squid_dataframe_to_array(iv_77_max)

### squid IV at 77K I_c min

iv_77_min = "06_01_2023-15_43_15_.csv"
I_77_min, V_77_min = squid_dataframe_to_array(iv_77_min)


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

x_line=np.linspace(70,120,1000)
x_line2=np.linspace(0,80,1000)

#param_77_min, cov_77_min = curve_fit(linear_fit, I_77_min, V_77_min)
#param_err_77_min = np.sqrt(np.diag(cov_77_min))




fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(I_77_max, V_77_max, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ maximized")
ax1.plot(I_77_min, V_77_min, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
ax1.plot(x_line, linear_fit(x_line,*param_Ic_max), linewidth = 2,linestyle="dashed", label=r"linear fit: $I_c$ max of the SQUID", color = 'green', zorder=2)
ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"I-V characteristic at 77 K, different flux offsets", fontsize = title_size)
ax1.set_xlim(0,max(I_77_max))
ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc=(0,0.5))
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Linear \ fit \ parameters: }$',
    r'm = %.2f$ \pm $ %.2f V/A' % (param_Ic_max[0], param_err_Ic_max[0]),
    r'q = %.2f $\pm %.2f \ \mu$V' % (param_Ic_max[1], param_err_Ic_max[1])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_iv_flux_offsets.png")

### Exchange axes ####

fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(V_77_max, I_77_max, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ maximized")
ax1.plot(V_77_min, I_77_min, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
#ax1.plot(x_line, linear_fit(x_line,*param_RT), linewidth = 1,linestyle="solid", label="linear fit: Ohm's law", color = 'green', zorder=2)
ax1.set_ylabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_xlabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"I-V characteristic at 77 K, different flux offsets", fontsize = title_size)
ax1.set_ylim(0,max(I_77_max))
ax1.set_xlim(0,max(V_77_max))
ax1.legend(loc="lower right")
ax1.grid()

fig1.savefig(f"{OUTDIR}/squid_iv_flux_offsets2.png")


print("----------------------------")
print("|       Squid @77          |")
print("|      deter. Ic max       |")
print("----------------------------")

print("\n Slope = %f +- %f Ohm\n" %(param_Ic_max[0], param_err_Ic_max[0]))
print("\n Intercept = %f +- %f  uV\n" %(param_Ic_max[1], param_err_Ic_max[1]))
Ic_max = -param_Ic_max[1]/param_Ic_max[0]
Ic_max_err = np.sqrt(pow(param_err_Ic_max[1]/param_Ic_max[0],2)+pow(param_Ic_max[1]*param_err_Ic_max[0]/(param_Ic_max[0]*param_Ic_max[0]),2))

print("\n Ic of the SQUID = %f +- %f  uA\n" %(Ic_max, Ic_max_err))

# critical current of JJ #

Ic1 = Ic_max/2
Ic1_err = Ic_max_err/2

print("\n Ic1 of the JJ = %f +- %f  uA\n" %(Ic1, Ic1_err))

# Josephson coupling energy #

phi_0 = scipy.constants.Planck/(2*scipy.constants.e)
Ej0 = Ic1 * phi_0 / (2*np.pi) ## uJ
Ej0_err = Ic1_err * phi_0 / (2*np.pi)

Ej0 = Ej0 *1e-6 / scipy.constants.e #eV
Ej0_err = Ej0_err *1e-6 / scipy.constants.e #eV

T = 77 # K
E_th = scipy.constants.Boltzmann*T/scipy.constants.e #eV
ratio = Ej0/E_th

print("\n Ej0 of the JJ = %f +- %f  eV\n" %(Ej0, Ej0_err))
print("\n Eth @ 77 K = %f eV\n" %(E_th))
print("\n Ratio = %f / %f = %f\n" %(Ej0, E_th, ratio))


##############################
####                      ####
####  NORMAL RESISTANCE   ####
####                      ####
##############################


### squid IV at 77K I_c max

iv_res = "06_01_2023-15_47_58_.csv"
I_res, V_res = squid_dataframe_to_array(iv_res)


##############################
####  OFFSET TRANSLATION  ####
##############################

V_res += V_offset
V_res += V_offset


## selecting data for linear fit ##

i_lin = I_res[np.abs(I_res) > 200 ]
v_lin = V_res[np.abs(I_res) > 200 ]

param_R_n, cov_R_n = curve_fit(linear_fit, i_lin, v_lin)
param_err_R_n = np.sqrt(np.diag(cov_R_n))

x_line=np.linspace(-400,400,4000)
#x_line2=np.linspace(0,80,1000)

#param_77_min, cov_77_min = curve_fit(linear_fit, I_77_min, V_77_min)
#param_err_77_min = np.sqrt(np.diag(cov_77_min))




fig1, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(I_res, V_res, linewidth = 1, linestyle = '-', label=r"I-V char")
#ax1.plot(i_lin, v_lin, linewidth = 1, linestyle = '-', label=r"I-V char, $I_c$ minimized")
ax1.plot(x_line, linear_fit(x_line,*param_R_n), linewidth = 2,linestyle="dashed", label=r"linear fit: Ohm's law for normal state", color = 'green', zorder=2)
#ax1.plot(x_line2, np.zeros(len(x_line)), linewidth = 2,linestyle="dashed", color = 'green', zorder=2)

ax1.set_xlabel(r'I [$\mu$A]', fontsize = label_size)
ax1.set_ylabel(r'V [$\mu$V]', fontsize = label_size)
ax1.set_title(r"I-V characteristic at 77 K, determining normal resistance", fontsize = title_size)
#ax1.set_xlim(0,max(I_77_max))
#ax1.set_ylim(-5,max(V_77_max))
ax1.legend(loc="lower right")
ax1.grid()

textstr = '\n'.join((
    r'$\bf{Linear \ fit \ parameters: }$',
    r'm = R = %.2f$ \pm $ %.2f $\Omega$' % (param_R_n[0], param_err_R_n[0]),
    r'q = %.2f $\pm %.2f \ \mu$V' % (param_R_n[1], param_err_R_n[1])))

props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


fig1.savefig(f"{OUTDIR}/squid_iv_normal_res.png")


print("\n\n")
print("----------------------------")
print("|       Squid @77          |")
print("|   deter. normal res      |")
print("----------------------------")

print("\n Slope = %f +- %f Ohm\n" %(param_R_n[0], param_err_R_n[0]))
print("\n Intercept = %f +- %f  uV\n" %(param_R_n[1], param_err_R_n[1]))
R_n = param_R_n[0]
R_n_err = param_err_R_n[0]

print("\n Normal resistance of the SQUID = %f +- %f Ohm\n" %(R_n, R_n_err))

## Normal resistance of JJ

R_n1 = R_n*2 
R_n1_err = R_n_err*2

print("\n Normal resistance of the JJ = %f +- %f  Ohm\n" %(R_n1, R_n1_err))


# Characteristic voltage

Vc = Ic_max*R_n
Vc_err = np.sqrt(pow(Ic_max_err*R_n,2)+pow(Ic_max*R_n_err,2))
Vc1 = Ic1*R_n1
Vc1_err = np.sqrt(pow(Ic1_err*R_n1,2)+pow(Ic1*R_n1_err,2))

print("\n Characteristic voltage of each JJ = %f +- %f  uV\n" %(Vc1, Vc1_err))
print("\n Characteristic voltage the SQUID = %f +- %f  uV\n" %(Vc, Vc_err))

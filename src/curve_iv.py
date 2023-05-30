import ROOT
import matplotlib.pyplot as plt
from utilities import *
from uncertainties import umath
import pandas as pd
import numpy as np
from scipy import constants as const


DATADIR = "../data"
OUTDIR = "../output"

#read dataframes iv-curves

         ######################
         ##                  ## 
         ##   I-V T=53.5K    ##
         ##                  ##
         ######################

file1 = "iv_48_0_2e-3_100pt"
I1, V1, Verr1 = iv_dataframe_to_array(file1)


file2 = "iv_51_5_1-5e-3_100pt"
I2, V2, Verr2 = iv_dataframe_to_array(file2)


file3 = "iv_53_5_1e-3_100pt"
I3, V3, Verr3 = iv_dataframe_to_array(file3)


file4 = "iv_54_5_1e-3_100pt"
I4, V4, Verr4 = iv_dataframe_to_array(file4)


file5 = "iv_55_5_1e-3_100pt"
I5, V5, Verr5 = iv_dataframe_to_array(file5)


file6 = "iv_56_5_1e-3_100pt"
I6, V6, Verr6 = iv_dataframe_to_array(file6)


file7 = "iv_57_5_1e-3_100pt"
I7, V7, Verr7 = iv_dataframe_to_array(file7)


file8 = "iv_62_0_1e-3_100pt"
I8, V8, Verr8 = iv_dataframe_to_array(file8)


## PLOTS

fig1, ax1 = plt.subplots(figsize=(8,6))

#ax1.errorbar(I1,V1, marker="o", linewidth = 0, markersize=3, label="T = 48.0 K")
#ax1.errorbar(I2,V2, marker="o", linewidth = 0, markersize=3, label="T = 51.5 K")
ax1.errorbar(I3,V3, marker="o", linewidth = 0, markersize=3, label="T = 53.5 K")
ax1.errorbar(I4,V4, marker="o", linewidth = 0, markersize=3, label="T = 54.5 K")
ax1.errorbar(I5,V5, marker="o", linewidth = 0, markersize=3, label="T = 55.5 K")
ax1.errorbar(I6,V6, marker="o", linewidth = 0, markersize=3, label="T = 56.5 K")
ax1.errorbar(I7,V7, marker="o", linewidth = 0, markersize=3, label="T = 57.5 K")
#ax1.errorbar(I8,V8, marker="o", linewidth = 0, markersize=3, label="T = 62.0 K")

ax1.set_xlabel('I[mA]')
ax1.set_ylabel('V[mV]')
ax1.set_title("I-V curves at different temperatures")
ax1.legend(loc="lower right")


fig1.savefig(f"{OUTDIR}/all_curves.png")

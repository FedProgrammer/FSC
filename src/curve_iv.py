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
print(V_err)



'''
fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.hist(deltaV_distribution, bins=30)
fig3.savefig(f"{OUTDIR}/deltaV_distribution.png")
'''

####### ALL CURVES ##########

fig1, ax1 = plt.subplots(figsize=(8,6))

#ax1.errorbar(I_48,V_48, marker="o", linewidth = 0, markersize=3, label="T = 48.0 K")
#ax1.errorbar(I_51,V_51, marker="o", linewidth = 0, markersize=3, label="T = 51.5 K")
ax1.errorbar(I_53,V_53, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 53.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'blue', mew = 0)
ax1.errorbar(I_54,V_54, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 54.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'orange', mew = 0)
ax1.errorbar(I_55,V_55, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 55.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'green', mew = 0)
ax1.errorbar(I_56,V_56, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 56.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'red', mew = 0)
ax1.errorbar(I_57,V_57, yerr= V_err, marker="o", linewidth = 0, markersize=3, label="T = 57.5 K", elinewidth = 0.5, ecolor = 'k', mfc = 'purple', mew = 0)
#ax1.errorbar(I_62,V_62, marker="o", linewidth = 0, markersize=3, label="T = 62.0 K")

ax1.set_xlabel('I[mA]', fontsize = label_size)
ax1.set_ylabel('V[mV]', fontsize = label_size)
ax1.set_title("I-V curves at different temperatures", fontsize = title_size)
ax1.legend(loc="lower right")
ax1.grid()
fig1.savefig(f"{OUTDIR}/all_curves.png")


T = [48.5, 51.5, 53.5, 54.5, 55.5, 56.5]
I_th = []
I_th_err = []
#G = scipy.constants.k*T/I_th
#G_err = []


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

def func_fit(x, y0, A, I_th):
    return y0 + A*np.sinh(x/I_th)

popt, pcov = curve_fit(func_fit, I_48, V_48, sigma = Verr_48, bounds=([-1e-1, 1e-4, 1e-4],[-1e-5, 1e-1, 1e-1]))
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_48, V_48, 'b-', label='data')
ax3.plot(I_48, func_fit(I_48, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")

'''
######### ODR ###########
def ODR_fit_func(p,x):
    return p[0]+p[1]*np.sinh(x/p[2])


fit_model = odr.Model(ODR_fit_func)
mydata = odr.RealData(I_48,V_48, sx=Ierr_48, sy=Verr_48)
myodr = odr.ODR(mydata, fit_model, beta0=[-1e-3, 1e-4, 1e-1])
myoutput = myodr.run()
myoutput.pprint()
'''




# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 48.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_48, I_48, V_48, Ierr_48, Verr_48)

fit_graph_err.SetTitle("I-V curve T = 48.5 K")
fit_graph_err.GetXaxis().SetTitle("I [mA]")
fit_graph_err.GetYaxis().SetTitle("V [mV]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit


function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1.5, 1.5)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.1, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 0.5)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_48_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))



####################
#### FIT 51.5 ######
####################


print("----------------------------")
print("|                          |")
print("|      Fit T = 51.5 K      |")
print("|                          |")
print("----------------------------")

ndata_51 = len(I_51)
Verr_51 = 10*np.ones(ndata_51)*V_err
Ierr_51 = np.zeros(ndata_51)

######## scipy optimize ######

def func_fit(x, y0, A, I_th):
    return y0 + A*np.sinh(x/I_th)

popt, pcov = curve_fit(func_fit, I_51, V_51, sigma = Verr_51)
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_51, V_51, 'b-', label='data')
ax3.plot(I_51, func_fit(I_51, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")


# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 51.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_51, I_51, V_51, Ierr_51, Verr_51)

fit_graph_err.SetTitle("I-V curve T = 51.5 K")
fit_graph_err.GetXaxis().SetTitle("V [mV]")
fit_graph_err.GetYaxis().SetTitle("I [mA]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit


function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1.5, 1.5)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.1, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 0.5)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_51_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))

####################
#### FIT 53.5 ######
####################


print("----------------------------")
print("|                          |")
print("|      Fit T = 53.5 K      |")
print("|                          |")
print("----------------------------")

ndata_53 = len(I_53)
Verr_53 = 10*np.ones(ndata_53)*V_err
Ierr_53 = np.zeros(ndata_53)

######## scipy optimize ######

def func_fit(x, y0, A, I_th):
    return y0 + A*np.sinh(x/I_th)

popt, pcov = curve_fit(func_fit, I_53, V_53, sigma = Verr_53)
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_53, V_53, 'b-', label='data')
ax3.plot(I_53, func_fit(I_53, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")


# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 53.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_53, I_53, V_53, Ierr_53, Verr_53)

fit_graph_err.SetTitle("I-V curve T = 53.5 K")
fit_graph_err.GetXaxis().SetTitle("V [mV]")
fit_graph_err.GetYaxis().SetTitle("I [mA]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit


function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1, 1)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.01, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 0.5)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_53_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))

####################
#### FIT 54.5 ######
####################

print("----------------------------")
print("|                          |")
print("|      Fit T = 54.5 K      |")
print("|                          |")
print("----------------------------")

ndata_54 = len(I_54)
Verr_54 = 10*np.ones(ndata_54)*V_err
Ierr_54 = np.zeros(ndata_54)

#scipy optimize

popt, pcov = curve_fit(func_fit, I_54, V_54, sigma = Verr_54)
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_54, V_54, 'b-', label='data')
ax3.plot(I_54, func_fit(I_54, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")

# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 54.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_54, I_54, V_54, Ierr_54, Verr_54)

fit_graph_err.SetTitle("I-V curve T = 54.5 K")
fit_graph_err.GetXaxis().SetTitle("V [mV]")
fit_graph_err.GetYaxis().SetTitle("I [mA]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit

function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1, 1)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.01, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 0.5)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_54_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))

####################
#### FIT 55.5 ######
####################

print("----------------------------")
print("|                          |")
print("|      Fit T = 55.5 K      |")
print("|                          |")
print("----------------------------")

ndata_55 = len(I_54)
Verr_55 = 10*np.ones(ndata_55)*V_err
Ierr_55 = np.zeros(ndata_55)

#scipy optimize

popt, pcov = curve_fit(func_fit, I_55, V_55, sigma = Verr_55)
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_55, V_55, 'b-', label='data')
ax3.plot(I_55, func_fit(I_55, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")

# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 55.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_55, I_55, V_55, Ierr_55, Verr_55)

fit_graph_err.SetTitle("I-V curve T = 55.5 K")
fit_graph_err.GetXaxis().SetTitle("V [mV]")
fit_graph_err.GetYaxis().SetTitle("I [mA]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit

function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1, 1)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.01, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 0.5)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_55_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))


####################
#### FIT 56.5 ######
####################

print("----------------------------")
print("|                          |")
print("|      Fit T = 56.5 K      |")
print("|                          |")
print("----------------------------")

ndata_56 = len(I_56)
Verr_56 = 10*np.ones(ndata_56)*V_err
Ierr_56 = np.zeros(ndata_56)

#scipy optimize

popt, pcov = curve_fit(func_fit, I_56, V_56, sigma = Verr_56)
print(popt)

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.plot(I_56, V_56, 'b-', label='data')
ax3.plot(I_56, func_fit(I_56, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
fig3.savefig(f"{OUTDIR}/fit.png")

# TGraphErrors

canvas = ROOT.TCanvas("c1", "curve fitting T = 56.5 K", 800, 600)
fit_graph_err = ROOT.TGraphErrors(ndata_56, I_56, V_56, Ierr_56, Verr_56)

fit_graph_err.SetTitle("I-V curve T = 56.5 K")
fit_graph_err.GetXaxis().SetTitle("V [mV]")
fit_graph_err.GetYaxis().SetTitle("I [mA]")
fit_graph_err.SetMarkerStyle(21)
fit_graph_err.SetMarkerSize(0.4)

# Fit

function_formula = "[0] + [1] * sinh(x/[2])"

fit_f = ROOT.TF1("fit1", function_formula, -1, 1)

#fit_f.setlinecolor(4)
fit_f.SetLineColor( 2 )
fit_f.SetLineWidth( 3 )
fit_f.SetMarkerColor( 4 )
fit_f.SetMarkerStyle( 21 )
fit_f.SetParName(0, "y_0")
fit_f.SetParName(1, "A")
fit_f.SetParName(2, "I_th")
#fit_f.SetParameter(0, - 0.001)
#fit_f.SetParameter(1, 0.001)
#fit_f.SetParameter(2, 1e-5)
fit_f.SetParLimits(0, -0.01, -0.001)
fit_f.SetParLimits(1, 0.0001, 0.1)
fit_f.SetParLimits(2, 0.0001, 1)

ROOT.gStyle.SetOptFit(1111111)

# Draw

canvas.Draw()
fit_graph_err.Fit(fit_f, "MSE")
fit_graph_err.Draw("AP")
canvas.SaveAs(f"{OUTPUTDIR}/fit_56_5.png")

print("Chi^2: %f, Probability: %f \n\n" %(fit_f.GetChisquare(), fit_f.GetProb()))

#ROOT.gApplication.Run()

I_th.append(fit_f.GetParameter(2))
I_th_err.append(fit_f.GetParError(2))



####################
#### I_th vs T #####
####################

print("----------------------------")
print("|                          |")
print("|      I_th vs Temper      |")
print("|                          |")
print("----------------------------")


fig4, ax4 = plt.subplots(figsize=(8,6))

ax4.errorbar(T, I_th, yerr=I_th_err, marker="o", linewidth = 0, markersize=3, label="")
ax4.set_xlabel('T [K]', fontsize = label_size)
ax4.set_ylabel('I_th [mA]', fontsize = label_size)
ax4.set_title("I_th vs T ", fontsize = title_size)
#ax4.legend(loc="lower right")
ax4.grid()
fig4.savefig(f"{OUTDIR}/I_th_vs_temp.png")

print(T)
print(I_th)


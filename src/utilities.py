import pandas as pd
import numpy as np

DATADIR = "../data"
DATADIR_SQUID = "../data/iv_RT"
DATADIR_SQUID2 = "../data/flux"
OUTPUTDIR = "../output"

def iv_dataframe_to_array(nomefile: str):
    data = pd.read_table(f"{DATADIR}/{nomefile}")
    col_name = list(data)
    I = np.array(data[col_name[0]]*1000) # mA
    V = np.array(data[col_name[1]]*1000) # mV
    V_err = V*0.01  
    return I, V, V_err


def rt_dataframe_to_array(nomefile: str):
    data = pd.read_table(f"{DATADIR}/{nomefile}")
    col_name = list(data)
    time = np.array(data[col_name[0]]) # tick
    T_diode = np.array(data[col_name[1]]) # K
    T_sample = np.array(data[col_name[2]]) # K
    V = np.array(data[col_name[3]]*1e6) # uV
    V_err = np.array(data[col_name[4]]*1e6) # uV
    #I = np.array(data[col_name[5]]*1e6) # uA
    R = np.array(data[col_name[6]]) # Ohm
    return time, T_diode, T_sample, V, V_err, R


def squid_dataframe_to_array(nomefile: str):
    data = pd.read_csv(f"{DATADIR_SQUID}/{nomefile}")
    col_name = list(data)
    current_bias = np.array(data[col_name[0]]) # uA
    voltage = np.array(data[col_name[1]]) # uV
    
    return current_bias, 
    
    
def squid_dataframe_to_array2(nomefile: str):
    data = pd.read_csv(f"{DATADIR_SQUID2}/{nomefile}")
    col_name = list(data)
    current_bias = np.array(data[col_name[0]]) # uA
    voltage = np.array(data[col_name[1]]) # uV
    
    return current_bias, voltage
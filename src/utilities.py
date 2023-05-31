import pandas as pd
import numpy as np

DATADIR = "../data"
OUTPUTDIR = "../output"

def iv_dataframe_to_array(nomefile: str):
    data = pd.read_table(f"{DATADIR}/{nomefile}")
    col_name = list(data)
    I = np.array(data[col_name[0]]*1000) # mA
    V = np.array(data[col_name[1]]*1000) # mV
    V_err = V*0.01  
    return I, V, V_err


    
import numpy as np
import pandas as pd
import os

f_name = 'face_data.csv'

def write(name,data):
    if os.path.isfile(f_name):
        
        df = pd.read_csv(f_name, index_col=0)
        
        latest = pd.DataFrame(data,columns=map(str,range(10000)))
        latest["name"] = name
        
        df = pd.concat((df,latest),ignore_index=True,sort=False)
        
        
    else:
        df = pd.DataFrame(data,columns=map(str,range(10000)))
        df['name'] = name
        
    df.to_csv(f_name)


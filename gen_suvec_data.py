import numpy as np
import pandas as pd

import random

c_act = 110
m_act = 1
c_dec = 100
m_dec = 3

possible_xes = [0,1,2,3,4,5,6,7,8]

nrows = 100000

x = np.random.choice(possible_xes, nrows)

dec = c_dec + m_dec*x
act = c_act + m_act*x

true_y = np.random.normal(act,act*0.1,nrows)
censor_y = np.random.normal(dec, dec*0.2, nrows)

dfDict = {'x':x, 'true_y':true_y, 'censor_y': censor_y}

df = pd.DataFrame(dfDict)

df['over'] = df['true_y']>df['censor_y']

df.to_csv('suvec.csv')

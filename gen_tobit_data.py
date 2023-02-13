import numpy as np
import pandas as pd

import random

c_u = 100
m_u = 1

c_p = 0.1
m_p = 0.01

censor_mean = 97
censor_sig = 0

possible_xes = [0,1,2,3,4,5,6,7]

nrows = 100000

x = np.random.choice(possible_xes, nrows)


u = c_u + m_u*x
p = c_p + m_p*x

true_y = np.random.normal(u,u*p,nrows)
censor_y = np.random.normal(censor_mean, censor_sig, nrows)

dfDict = {'x':x, 'true_y':true_y, 'censor_y': censor_y}

df = pd.DataFrame(dfDict)

df['y'] = df[['true_y','censor_y']].min(axis=1)

df['censored'] = df['censor_y']<df['true_y']

df.to_csv('gnormal.csv')

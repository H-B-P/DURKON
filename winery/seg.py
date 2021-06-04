import pandas as pd
import numpy as np
import math
import util

def get_segpt(df, col, ratio, iters=30, sq=True):
 if sq:
  trueRatio=ratio*ratio
 else:
  trueRatio=ratio
 ub = max(df[col])
 lb = min(df[col])
 for i in range(iters):
  mp = (ub+lb)/2.0
  #print(lb, mp, ub)
  upperSlice = df[df[col]>mp][col] - mp
  lowerSlice = mp - df[df[col]<mp][col]
  upperSum = sum(upperSlice)
  lowerSum = sum(lowerSlice)
  if lowerSum<=0: #avoid div by zero
   lb=mp
  else:
   if (upperSum/lowerSum)>trueRatio:
    lb=mp
   else:
    ub=mp
 return (ub+lb)/2.0

def get_ratios(nseg):
 ratios=[]
 for i in range(1,nseg):
  ratios.append(float(i)/float(nseg-i))
 return ratios

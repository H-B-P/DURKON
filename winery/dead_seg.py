import pandas as pd
import numpy as np
import math
import util

def gimme_pseudo_winsors(inputDf, col, pw=0.05):
 return util.round_to_sf(inputDf[col].quantile(pw),3), util.round_to_sf(inputDf[col].quantile(1-pw),3)

def gimme_starting_affect(inputDf, col, segs):
 x = inputDf[col]
 x1 = float(segs[0])
 x2 = float(segs[1])
 affectedness = pd.Series([0]*len(inputDf))
 affectedness.loc[(x<x1)] = 1
 affectedness.loc[(x>=x1) & (x<x2)] = (x2 - x)/(x2 - x1)
 return sum(affectedness)

def gimme_normie_affect(inputDf, col, segs, posn):
 x = inputDf[col]
 x1 = float(segs[posn-1])
 x2 = float(segs[posn])
 x3 = float(segs[posn+1])
 affectedness = pd.Series([0]*len(inputDf))
 affectedness.loc[(x>=x1) & (x<x2)] = (x - x1)/(x2 - x1)
 affectedness.loc[(x>=x2) & (x<x3)] = (x3 - x)/(x3 - x2)
 return sum(affectedness)

def gimme_ending_affect(inputDf, col, segs):
 x = inputDf[col]
 x1 = float(segs[-2])
 x2 = float(segs[-1])
 affectedness = pd.Series([0]*len(inputDf))
 affectedness.loc[(x>=x2)] = 1
 affectedness.loc[(x>=x1) & (x<x2)] = (x - x1)/(x2 - x1)
 return sum(affectedness)

def gimme_sa_optimizing_func(inputDf, col, segsSoFar):
 def sa_optimizing_func(x):
  return gimme_starting_affect(inputDf, col, segsSoFar+[x])
 return sa_optimizing_func

def gimme_na_optimizing_func(inputDf, col, segsSoFar):
 def na_optimizing_func(x):
  return gimme_normie_affect(inputDf, col, segsSoFar+[x], len(segsSoFar)-1)
 return na_optimizing_func

def gimme_pa_optimizing_func(inputDf, col, segsSoFar, end):
 def pa_optimizing_func(x):
  return gimme_normie_affect(inputDf, col, segsSoFar+[x]+[end], len(segsSoFar))
 return pa_optimizing_func

if __name__ == "__main__":
 dyct = {"x":list(range(100))}
 df=pd.DataFrame(dyct)
 start, end = gimme_pseudo_winsors(df, "x")
 print(start, end)

 targetLen=5
 goodAmt=float(len(df))/targetLen
 segs = [start]
 print(segs)
 if targetLen>2:
  optFunc = gimme_sa_optimizing_func(df, "x", segs)
  next = util.target_input_with_output(optFunc, goodAmt, start, end)
  segs.append(util.round_to_sf(next,3))
  print(segs)
 for i in range(targetLen-3):
  optFunc = gimme_na_optimizing_func(df, "x", segs)
  next = util.target_input_with_output(optFunc, goodAmt, start, end)
  segs.append(util.round_to_sf(next,3))
  print(segs)
 segs.append(end)
 print(segs)
 print([gimme_starting_affect(df, "x", segs), gimme_normie_affect(df, "x", segs, 1), gimme_normie_affect(df, "x", segs, 2), gimme_normie_affect(df, "x", segs, 3), gimme_ending_affect(df, "x", segs)])

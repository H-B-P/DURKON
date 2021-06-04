import pandas as pd
import numpy as np
import math
import copy

def apply_pena(value, pena, boringValue=1):
 if value>boringValue:
  return max(boringValue, value-pena)
 else:
  return min(boringValue, value+pena)

def penalize_this_cont_col(model, col, penalties={"contStraight":0.01, "contGroup":0.01}, boringValue=1):
 newModel=copy.deepcopy(model)
 if "contStraight" in penalties:
  for i in range(len(model["conts"][col])):
   newModel["conts"][col][i][1] = apply_pena(newModel["conts"][col][i][1], penalties["contStraight"], boringValue)
 if "contGroup" in penalties:
  sumofSq = 0
  for i in range(len(model["conts"][col])):
   sumofSq += (model["conts"][col][i][1]-boringValue)**2
  if sumofSq>0:
   for i in range(len(model["conts"][col])):
    newModel["conts"][col][i][1] = apply_pena(newModel["conts"][col][i][1], penalties["contGroup"]*abs(model["conts"][col][i][1]-boringValue)/math.sqrt(sumofSq))
 return newModel

def flatten_this_cont_col(model, col, penalties={"contFlatten":0.01}):
 newModel=copy.deepcopy(model)
 if "flatten" in penalties:
  for i in range(len(model["conts"][col])-1):
   newModel["conts"][col][i][1] += penalties["contFlatten"]*(model["conts"][col][i+1][1]-model["conts"][col][i][1])
   newModel["conts"][col][i+1][1] += penalties["contFlatten"]*(model["conts"][col][i][1]-model["conts"][col][i+1][1])
 return newModel

if __name__ == '__main__':
 model = {"BIG_C":1.0,"conts":{"x":[[0,0],[1,1]]}, "cats":[]}
 print(penalize_this_cont_col(model,"x", {"contStraight":0.01}))
 model = {"BIG_C":1.0,"conts":{"x":[[0,0],[1,1]]}, "cats":[]}
 print(penalize_this_cont_col(model,"x", {"contGroup":0.01}))
 

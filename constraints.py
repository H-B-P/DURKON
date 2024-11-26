import pandas as pd
import numpy as np

import copy

#minimum relativities; useful for preventing multipliers from going <0

def get_enforce_min_rela(minRela=0.1):
 
 def enforce_min_rela(model): #Not expanded to interxes 
  
  opModel = copy.deepcopy(model)
  
  if "conts" in opModel:
   for col in opModel["conts"]:
    for i in range(len(opModel["conts"][col])):
     opModel["conts"][col][i][1] = max(minRela, opModel["conts"][col][i][1])
  
  if "cats" in opModel:
   for col in opModel["cats"]:
    for u in opModel["cats"][col]["uniques"]:
     opModel["cats"][col]["uniques"][u] = max(minRela, opModel["cats"][col]["uniques"][u])
    opModel["cats"][col]["OTHER"] = max(minRela, opModel["cats"][col]["OTHER"])
  
  return opModel
 
 return enforce_min_rela


#Monotony

def gently_monotonize_this_list(l, increasing=True, r=10):
 
 newL = l.copy()
 
 curseList = [False]*len(l)
 for i in range(len(newL)-1):
  if ((l[i+1]<l[i]) and increasing) or ((l[i+1]>l[i]) and not increasing):
   curseList[i]=True
   curseList[i+1]=True
 
 i = 0
 while i < len(l):
  if curseList[i]:
   start = i
   while (i < len(newL)) and (curseList[i]):
    i += 1
   end = i
   avg = sum(newL[start:end]) / (end - start)
   for j in range(start, end):
    newL[j] = avg
  else:
   i += 1
 
 if r<=1:
  return newL
 else:
  return gently_monotonize_this_list(newL, increasing, r-1)

def firmly_monotonize_this_list(l, increasing=True):
 newL = l.copy()
 if increasing:
  for i in range(len(newL)-1):
   if newL[i+1]<newL[i]:
    newL[i+1]=newL[i]
 else:
  for i in range(len(newL)-1):
   if newL[i+1]>newL[i]:
    newL[i+1]=newL[i]
 return newL

def monotonize_this_list(l, increasing=True, r=10):
 return firmly_monotonize_this_list(gently_monotonize_this_list(l, increasing, r),increasing)

def monotonize_this_cat(model, col, increasing, r=10):
 newModel = copy.deepcopy(model)
 
 k = list(model["cats"][col]["uniques"].keys())
 v = list(model["cats"][col]["uniques"].values())
 newv = monotonize_this_list(v, increasing, r)
 
 for i in range(len(newv)):
  newModel["cats"][col]["uniques"][k[i]]=newv[i]
 
 return newModel

def monotonize_this_cont(model, col, increasing, r=10):
 newModel = copy.deepcopy(model)
 
 v = [x[1] for x in model["conts"][col]]
 newv = monotonize_this_list(v, increasing, r)
 
 for i in range(len(newv)):
  newModel["conts"][col][i][1]=newv[i]
 
 return newModel

def get_monotonize_this_model(catIncDict, contIncDict, r=10):
 def monotonize_this_model(model):
  newModel = copy.deepcopy(model)
  for col in catIncDict:
   newModel = monotonize_this_cat(newModel, col, catIncDict[col], r)
  for col in contIncDict:
   newModel = monotonize_this_cont(newModel, col, contIncDict[col], r)
  return newModel
 return monotonize_this_model

#todo: variant which enforces Jake's Law on unknowns

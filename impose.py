import pandas as pd
import numpy as np

import copy

#Drag to default

def move_to_default(x, l, defaultValue=1):
 if x>defaultValue:
  return max(x-l, defaultValue)
 if x<defaultValue:
  return min(x+l, defaultValue)
 return x

def get_penalize_nondefault(pen, defaultValue=1, specificPenas={}):
 
 def penalize_nondefault(model):
  
  print("OMG HAIII!")
  if "cats" in model:
   for cat in model["cats"]:
   
    if cat in specificPenas:
     l = specificPenas[cat]
    else:
     l = pen
    
    for u in model["cats"][cat]["uniques"]:
     model["cats"][cat]["uniques"][u] = move_to_default(model["cats"][cat]["uniques"][u], l, defaultValue)
    model["cats"][cat]["OTHER"] = move_to_default(model["cats"][cat]["OTHER"], l, defaultValue)
  
  if "conts" in model:
   for cont in model["conts"]:
    
    if cont in specificPenas:
     l = specificPenas[cont]
    else:
     l = pen
    
    for i in range(len(model["conts"][cont])):
     model["conts"][cont][i][1] = move_to_default(model["conts"][cont][i][1], l, defaultValue)
  
  if "catcats" in model:
   for catcat in model["catcats"]:
    
    if catcat in specificPenas:
     l = specificPenas[catcat]
    else:
     l = pen
    
    for u1 in model["catcats"][catcat]["uniques"]:
     model["catcats"][catcat]["uniques"][u1]["OTHER"] = move_to_default(model["catcats"][catcat]["uniques"][u1]["OTHER"], l, defaultValue)
     for u2 in model["catcats"][catcat]["uniques"][u1]["uniques"]:
      model["catcats"][catcat]["uniques"][u1]["uniques"][u2] = move_to_default(model["catcats"][catcat]["uniques"][u1]["uniques"][u2], l, defaultValue)
    for u2 in model["catcats"][catcat]["OTHER"]["uniques"]:
     model["catcats"][catcat]["OTHER"]["uniques"][u2] = move_to_default(model["catcats"][catcat]["OTHER"]["uniques"][u2], l, defaultValue)
    model["catcats"][catcat]["OTHER"]["OTHER"]=move_to_default(model["catcats"][catcat]["OTHER"]["OTHER"], l, defaultValue)
  
  if "catconts" in model:
   for catcont in model["catconts"]:
    
    if catcont in specificPenas:
     l = specificPenas[catcont]
    else:
     l = pen
    
    for u in model["catconts"][catcont]["uniques"]:
     for i in range(len(model["catconts"][catcont]["uniques"][u])):
      model["catconts"][catcont]["uniques"][u][i][1] = move_to_default(model["catconts"][catcont]["uniques"][u][i][1], l, defaultValue)
    for i in range(len(model["catconts"][catcont]["OTHER"])):
     model["catconts"][catcont]["OTHER"][i][1] = move_to_default(model["catconts"][catcont]["OTHER"][i][1], l, defaultValue)
  
  if "contconts" in model:
   for contcont in model["contconts"]:
    
    if contcont in specificPenas:
     l = specificPenas[contcont]
    else:
     l = pen
    
    for i in range(len(model["contconts"][contcont])):
     for j in range(len(model["contconts"][contcont][i][1])):
      model["contconts"][contcont][i][1][j][1] = move_to_default(model["contconts"][contcont][i][1][j][1], l, defaultValue)
  
  return model
 
 return penalize_nondefault

#Combat continuous complexity

def move_to_target(x, l, t):
 if (x<t):
  return min(x+l, x+(t-x)/2)
 if (x>t):
  return max(x-l, x-(x-t)/2)
 return x

def penalize_cont_complexity(cont, pen):
 if (len(cont)>2):
  targets = []
  targets.append(cont[1][1] - (cont[2][1]-cont[1][1])/(cont[2][0]-cont[1][0])*(cont[1][0]-cont[0][0]))
  for i in range(len(cont)-2):
   targets.append(cont[i][1] + (cont[i+2][1]-cont[i][1])/(cont[i+2][0]-cont[i][0])*(cont[i+1][0]-cont[i][0]))
  targets.append(cont[-2][1] + (cont[-2][1]-cont[-3][1])/(cont[-2][0]-cont[-3][0])*(cont[-1][0]-cont[-2][0]))
  
  for i in range(len(cont)):
   cont[i][1] = move_to_target(cont[i][1], pen, targets[i])
  
  return cont

def get_penalize_conts_complexity(pen):
 def penalize_conts_complexity(model):
  for c in model["conts"]:
   model["conts"][c] = penalize_cont_complexity(model["conts"][c], pen)
  return model
 return penalize_conts_complexity

#Looping conts

def get_enforce_loops(cols):
 
 def enforce_loops(model):
  opModel = copy.deepcopy(model)
  for c in cols:
   opModel["conts"][c][0][1] = (model["conts"][c][0][1] + model["conts"][c][-1][1])/2
   opModel["conts"][c][-1][1] = (model["conts"][c][0][1] + model["conts"][c][-1][1])/2
  return opModel
 
 return enforce_loops

#Minimum relativities; useful for preventing multipliers from going <0

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


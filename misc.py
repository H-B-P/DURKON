import pandas as pd
import numpy as np
import math
import copy
import time

import util
import calculus
import rele
import prep

def what_cats(model):
 if "cats" in model:
  return [c for c in model["cats"]]
 else:
  return []

def what_conts(model):
 if "conts" in model:
  return [c for c in model["conts"]]
 else:
  return []

def how_many_cats(model):
 return len(what_cats(model))

def how_many_conts(model):
 return len(what_conts(model))


def de_feat(model, defaultValue=1): #Not expanded to interxes
 oldModel=copy.deepcopy(model)
 newModel={"BASE_VALUE":oldModel["BASE_VALUE"], "conts":{}, "cats":{}}
 if "featcomb" in oldModel:
  newModel["featcomb"]=oldModel["featcomb"]
 
 if "conts" in oldModel:
  for col in oldModel["conts"]:
   empty=True
   for pt in oldModel["conts"][col]:
    if pt[1]!=defaultValue:
     empty=False
   if not empty:
    newModel["conts"][col]=oldModel["conts"][col]
 
 if "cats" in oldModel:
  for col in oldModel["cats"]:
   empty=True
   if oldModel["cats"][col]["OTHER"]!=defaultValue:
    empty=False
   for unique in oldModel["cats"][col]["uniques"]:
    if oldModel["cats"][col]["uniques"][unique]!=defaultValue:
     empty=False
   if not empty:
    newModel["cats"][col]=oldModel["cats"][col]
 
 return newModel

def get_sorted_keys(cat):
 keys = [c for c in cat["uniques"]]
 keys.sort()
 return keys

def get_effect_of_this_cont_col_from_relevances(reles, model, col, defaultValue=1):
 postmultmat = np.array([pt[1] for pt in model["conts"][col]]+[defaultValue])
 return np.matmul(reles,postmultmat)

def get_effects_of_cont_cols_from_relevance_dict(releDict, model):
 opDict = {}
 if "conts" in model:
  for col in model["conts"]:
   opDict[col]= get_effect_of_this_cont_col_from_relevances(releDict[col], model, col)
 return opDict
 
def get_effect_of_this_cat_col_from_relevances(reles, model, col):
 skeys = get_sorted_keys(model['cats'][col])
 postmultmat = np.array([model["cats"][col]["uniques"][key] for key in skeys]+[model["cats"][col]["OTHER"]])
 return np.matmul(reles,postmultmat)
 
def get_effects_of_cat_cols_from_relevance_dict(releDict, model):
 opDict = {}
 if "cats" in model:
  for col in model["cats"]:
   opDict[col]= get_effect_of_this_cat_col_from_relevances(releDict[col], model, col)
 return opDict

def get_effect_of_this_catcat_from_relevances(reles, model, cols):
 skeys1 = get_sorted_keys(model['catcats'][cols])
 skeys2 = get_sorted_keys(model['catcats'][cols]["OTHER"])
 postmultmat = []
 for key1 in skeys1:
  postmultmat = postmultmat+[model['catcats'][cols]['uniques'][key1]['uniques'][key2] for key2 in skeys2]+[model['catcats'][cols]['uniques'][key1]['OTHER']]
 postmultmat = postmultmat + [model['catcats'][cols]['OTHER']['uniques'][key2] for key2 in skeys2]+ [model['catcats'][cols]['OTHER']['OTHER']]
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effect_of_this_catcont_from_relevances(reles, model, cols, defaultValue=1):
 skeys = get_sorted_keys(model["catconts"][cols])
 postmultmat = []
 for key in skeys:
  postmultmat = postmultmat + [pt[1] for pt in model["catconts"][cols]['uniques'][key]]+[defaultValue]
 postmultmat = postmultmat + [pt[1] for pt in model["catconts"][cols]['OTHER']] + [defaultValue]
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effect_of_this_contcont_from_relevances(reles, model, cols, defaultValue=1):
 postmultmat = []
 for pt1 in model['contconts'][cols]:
  postmultmat = postmultmat + [pt2[1] for pt2 in pt1[1]] + [defaultValue]
 postmultmat = postmultmat + [defaultValue]*(len(model['contconts'][cols][-1][1])+1)
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effects_of_interxns_from_relevance_dict(releDict, model):
 opDict = {}
 if "catcats" in model:
  for cols in model['catcats']:
   opDict[cols] = get_effect_of_this_catcat_from_relevances(releDict[cols],model, cols)
 if "catconts" in model:
  for cols in model['catconts']:
   opDict[cols] = get_effect_of_this_catcont_from_relevances(releDict[cols],model,cols)
 if "contconts" in model:
  for cols in model['contconts']:
   opDict[cols] = get_effect_of_this_contcont_from_relevances(releDict[cols],model,cols)
 return opDict

def comb_from_effects_addl(base,l,contEffs,catEffs,interxEffs=None):
 op = pd.Series([base]*l)
 for col in contEffs:
  op = op+contEffs[col]
 for col in catEffs:
  op = op+catEffs[col]
 if interxEffs!=None:
  for cols in interxEffs:
   op = op+interxEffs[cols]
 return op

def comb_from_effects_mult(base,l,contEffs,catEffs,interxEffs=None):
 op = pd.Series([base]*l)
 for col in contEffs:
  op = op*contEffs[col]
 for col in catEffs:
  op = op*catEffs[col]
 if interxEffs!=None:
  for cols in interxEffs:
   op = op*interxEffs[cols]
 return op

def get_effect_of_this_cont_on_single_input(x, cont):
 if x<=cont[0][0]:
  return cont[0][1] #everything outside our scope is flat
 for i in range(len(cont)-1):
  if (x>=cont[i][0] and x<=cont[i+1][0]):
   return ((x-cont[i][0])*cont[i+1][1] + (cont[i+1][0]-x)*cont[i][1])/(cont[i+1][0]-cont[i][0]) #((x-p1)y1 + (p2-x)y2) / (p2 - p1)
 if x>=cont[-1][0]:
  return cont[-1][1] #everything outside our scope is flat
 return "idk lol"

def get_effect_of_this_cont_col(inputDf, model, col):
 x = inputDf[col]
 effectOfCol = pd.Series([1]*len(inputDf))
 effectOfCol.loc[(x<=model["conts"][col][0][0])] = model["conts"][col][0][1] #Everything too early gets with the program
 for i in range(len(model["conts"][col])-1):
  x1 = model["conts"][col][i][0]
  x2 = model["conts"][col][i+1][0]
  y1 = model["conts"][col][i][1]
  y2 = model["conts"][col][i+1][1]
  effectOfCol.loc[(x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[x>=model["conts"][col][-1][0]] = model["conts"][col][-1][1] #Everything too late gets with the program
 return effectOfCol

def get_effect_of_this_cat_on_single_input(x, cat): #slightly roundabout approach so we can copy for columns
 for unique in cat["uniques"]:
  if x==unique:
   return cat["uniques"][unique]
 return cat["OTHER"]

def get_effect_of_this_cat_col(inputDf, model, col):
 effectOfCol = pd.Series([model["cats"][col]["OTHER"]]*len(inputDf))
 for unique in model["cats"][col]["uniques"]:
  effectOfCol[inputDf[col]==unique] = model["cats"][col]["uniques"][unique]
 return effectOfCol

def get_effect_of_this_catcat_on_single_input(x1, x2, catcat):
 for unique in catcat["uniques"]:
  if x1==unique:
   return get_effect_of_this_cat_on_single_input(x2, catcat["uniques"][unique])
 return get_effect_of_this_cat_on_single_input(x2, catcat["OTHER"])

def get_effect_of_this_catcat(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 effectOfCol = pd.Series([model["catcats"][cols]["OTHER"]["OTHER"]]*len(inputDf))
 for unique1 in model['catcats'][cols]['uniques']:
  for unique2 in model['catcats'][cols]['uniques'][unique1]['uniques']:
   effectOfCol[(inputDf[col1]==unique1) & (inputDf[col2]==unique2)] = model['catcats'][cols]['uniques'][unique1]['uniques'][unique2]
  effectOfCol[(inputDf[col1]==unique1) & (~inputDf[col2].isin(model['catcats'][cols]['uniques'][unique1]['uniques']))] = model['catcats'][cols]['uniques'][unique1]['OTHER']
 for unique2 in model['catcats'][cols]['OTHER']['uniques']:
  effectOfCol[(~inputDf[col1].isin(model['catcats'][cols]['uniques'])) & (inputDf[col2]==unique2)] = model['catcats'][cols]['OTHER']['uniques'][unique2]
 return effectOfCol

def get_effect_of_this_catcont_on_single_input(x1, x2, catcont):
 for unique in catcont["uniques"]:
  if x1==unique:
   return get_effect_of_this_cont_on_single_input(x2, catcont["uniques"][unique])
 return get_effect_of_this_cont_on_single_input(x2, catcont["OTHER"])

def get_effect_of_this_catcont(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 x = inputDf[col2]
 effectOfCol = pd.Series([1]*len(inputDf))
 
 for unique in model['catconts'][cols]['uniques']:
  effectOfCol.loc[(inputDf[col1]==unique) & (x<=model["catconts"][cols]['uniques'][unique][0][0])] = model["catconts"][cols]['uniques'][unique][0][1] #Everything too early gets with the program
  for i in range(len(model["catconts"][cols]['uniques'][unique])-1):
   x1 = model["catconts"][cols]['uniques'][unique][i][0]
   x2 = model["catconts"][cols]['uniques'][unique][i+1][0]
   y1 = model["catconts"][cols]['uniques'][unique][i][1]
   y2 = model["catconts"][cols]['uniques'][unique][i+1][1]
   effectOfCol.loc[(inputDf[col1]==unique) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
  effectOfCol.loc[(inputDf[col1]==unique) & (x>=model["catconts"][cols]['uniques'][unique][-1][0])] = model["catconts"][cols]['uniques'][unique][-1][1] #Everything too late gets with the program
  
 effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x<=model["catconts"][cols]['OTHER'][0][0])] = model["catconts"][cols]['OTHER'][0][1] #Everything too early gets with the program
 for i in range(len(model["catconts"][cols]['OTHER'])-1):
  x1 = model["catconts"][cols]['OTHER'][i][0]
  x2 = model["catconts"][cols]['OTHER'][i+1][0]
  y1 = model["catconts"][cols]['OTHER'][i][1]
  y2 = model["catconts"][cols]['OTHER'][i+1][1]
  effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x>=model["catconts"][cols]['OTHER'][-1][0])] = model["catconts"][cols]['OTHER'][-1][1] #Everything too late gets with the program
 
 return effectOfCol

#this next one is ludicrously inefficient but if you're using it in a context where that matters you made some bad life choices somewhere
def get_effect_of_this_contcont_on_single_input(x1,x2, contcont):
 x1cont = [[c[0], get_effect_of_this_cont_on_single_input(x2, c[1])] for c in contcont]
 return get_effect_of_this_cont_on_single_input(x1, x1cont)


def get_effect_of_this_contcont(inputDf,model,cols): #we are using x and y to predict z
 col1, col2 = cols.split(' X ')
 x = inputDf[col1]
 y = inputDf[col2]
 effectOfCol = pd.Series([1]*len(inputDf))
 
 #Corners get with the program
 
 effectOfCol.loc[(x<=model["contconts"][cols][0][0]) & (y<=model["contconts"][cols][0][1][0][0])] = model["contconts"][cols][0][1][0][1]
 effectOfCol.loc[(x>=model["contconts"][cols][-1][0]) & (y<=model["contconts"][cols][-1][1][0][0])] = model["contconts"][cols][-1][1][0][1]
 effectOfCol.loc[(x<=model["contconts"][cols][0][0]) & (y>=model["contconts"][cols][0][1][-1][0])] = model["contconts"][cols][0][1][-1][1]
 effectOfCol.loc[(x>=model["contconts"][cols][-1][0]) & (y>=model["contconts"][cols][-1][1][-1][0])] = model["contconts"][cols][-1][1][-1][1]
 
 #Edges get with the program
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  z1 = model["contconts"][cols][i][1][0][1]
  z2 = model["contconts"][cols][i+1][1][0][1]
  effectOfCol.loc[(y<=model["contconts"][cols][i][1][0][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  z1 = model["contconts"][cols][i][1][-1][1]
  z2 = model["contconts"][cols][i+1][1][-1][1]
  effectOfCol.loc[(y>=model["contconts"][cols][i][1][-1][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(model["contconts"][cols][0][1])-1):
  y1 = model["contconts"][cols][0][1][i][0]
  y2 = model["contconts"][cols][0][1][i+1][0]
  z1 = model["contconts"][cols][0][1][i][1]
  z2 = model["contconts"][cols][0][1][i+1][1]
  effectOfCol.loc[(x<=model["contconts"][cols][0][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 for i in range(len(model["contconts"][cols][-1][1])-1):
  y1 = model["contconts"][cols][-1][1][i][0]
  y2 = model["contconts"][cols][-1][1][i+1][0]
  z1 = model["contconts"][cols][-1][1][i][1]
  z2 = model["contconts"][cols][-1][1][i+1][1]
  effectOfCol.loc[(x>=model["contconts"][cols][-1][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 #The interior
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  for j in range(len(model["contconts"][cols][i][1])-1):
   y1 = model["contconts"][cols][0][1][j][0]
   y2 = model["contconts"][cols][0][1][j+1][0]
   z11 = model["contconts"][cols][i][1][j][1]
   z12 = model["contconts"][cols][i][1][j+1][1]
   z21 = model["contconts"][cols][i+1][1][j][1]
   z22 = model["contconts"][cols][i+1][1][j+1][1]
   effectOfCol.loc[(x>=x1)&(x<=x2)&(y>=y1)&(y<=y2)] = ((x-x1)*(y-y1)*z22 + (x-x1)*(y2-y)*z21 + (x2-x)*(y-y1)*z12 + (x2-x)*(y2-y)*z11)/((x2 - x1)*(y2 - y1))
 
 return effectOfCol

def lambdapply(func, slist):
 ldf = pd.DataFrame()
 for i in range(len(slist)):
  ldf[i] = slist[i]
 return ldf.apply(lambda x: func(*[x[c] for c in range(len(slist))]), axis=1)

def predict_mult(inputDf, model):
 preds = pd.Series([model["BASE_VALUE"]]*len(inputDf))
 if "conts" in model:
  for col in model["conts"]:
   effectOfCol = get_effect_of_this_cont_col(inputDf, model, col)
   preds = preds*effectOfCol
 if "cats" in model:
  for col in model["cats"]:
   effectOfCol = get_effect_of_this_cat_col(inputDf, model, col)
   preds = preds*effectOfCol
 if "catcats" in model:
  for cols in model["catcats"]:
   effectOfCol = get_effect_of_this_catcat(inputDf, model, cols)
   preds = preds*effectOfCol
 if "catconts" in model:
  for cols in model["catconts"]:
   effectOfCol = get_effect_of_this_catcont(inputDf, model, cols)
   preds = preds*effectOfCol
 if "contconts" in model:
  for cols in model["contconts"]:
   effectOfCol = get_effect_of_this_contcont(inputDf, model, cols)
   preds = preds*effectOfCol
 
 return preds

def predict_addl(inputDf, model):
 preds = pd.Series([model["BASE_VALUE"]]*len(inputDf))
 if "conts" in model:
  for col in model["conts"]:
   effectOfCol = get_effect_of_this_cont_col(inputDf, model, col)
   preds = preds+effectOfCol
 if "cats" in model:
  for col in model["cats"]:
   effectOfCol = get_effect_of_this_cat_col(inputDf, model, col)
   preds = preds+effectOfCol
 if "catcats" in model:
  for cols in model["catcats"]:
   effectOfCol = get_effect_of_this_catcat(inputDf, model, cols)
   preds = preds+effectOfCol
 if "catconts" in model:
  for cols in model["catconts"]:
   effectOfCol = get_effect_of_this_catcont(inputDf, model, cols)
   preds = preds+effectOfCol
 if "contconts" in model:
  for cols in model["contconts"]:
   effectOfCol = get_effect_of_this_contcont(inputDf, model, cols)
   preds = preds+effectOfCol
 
 return preds

def predict_model(inputDf, model, linkage = "Unity"):
 if "featcomb" in model:
  if model["featcomb"]=="addl":
   comb = predict_addl(inputDf, model)
  else:
   comb = predict_mult(inputDf, model)
 else:
  comb = predict_mult(inputDf, model)
 
 if type(linkage)==str:
  return comb.apply(calculus.links[linkage])
 else:
  return comb.apply(linkage)

def predict_models(inputDf, models, linkage=calculus.Add_mlink):
 combs = []
 for model in models:
  combs.append(predict_model(inputDf, model))
 
 if type(linkage)==str:
  return lambdapply(calculus.links[linkage], combs)
 else:
  return lambdapply(linkage, combs)

def predict(inputDf, predictor, linkage=None):
 if type(predictor)==dict:
  if linkage==None:
   return predict_model(inputDf, predictor)
  return predict_model(inputDf, predictor, linkage)
 if type(predictor)==list:
  if linkage==None:
   return predict_models(inputDf, predictor)
  return predict_models(inputDf, predictor, linkage)

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def roundify_cat(cat, sf=5):
 op=cat.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = round(op[k][unique], sf)
  else:
   op[k]=round(op[k], sf)
 return op

def roundify_cont(cont, sf=5):
 op = copy.deepcopy(cont)
 for i in range(len(op)):
  op[i][1] = round(op[i][1],sf)
 return op

def roundify_catcat(catcat, sf=5):
 op=catcat.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = roundify_cat(op[k][unique], sf)
  else:
   op[k]=roundify_cat(op[k], sf)
 return op

def roundify_catcont(catcont, sf=5):
 op=catcont.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = roundify_cont(op[k][unique], sf)
  else:
   op[k]=roundify_cont(op[k], sf)
 return op

def roundify_contcont(contcont, sf=5):
 op = copy.deepcopy(contcont)
 for i in range(len(op)):
  op[i][1] = roundify_cont(op[i][1],sf)
 return op


def explain(model, sf=5):
 print("BASE_VALUE", round_to_sf(model["BASE_VALUE"], sf))
 if "cats" in model:
  for col in model["cats"]:
   print(col, roundify_cat(model["cats"][col], sf))
 if "conts" in model:
  for col in model["conts"]:
   print(col, roundify_cont(model["conts"][col], sf))
 if "catcats" in model:
  for cols in model["catcats"]:
   print(cols, roundify_catcat(model["catcats"][cols], sf))
 if "catconts" in model:
  for cols in model["catconts"]:
   print(cols, roundify_catcont(model["catconts"][cols], sf))
 if "contconts" in model:
  for cols in model["contconts"]:
   print(cols, roundify_contcont(model["contconts"][cols], sf))
 print("-")


def normalize_model(model, totReleDict): #Not expanded to interxes
 
 opModel = copy.deepcopy(model)
 
 for col in totReleDict["conts"]:
  relaTimesRele = 0
  for i in range(len(opModel["conts"][col])):
   relaTimesRele += opModel["conts"][col][i][1] * totReleDict["conts"][col][i]
  averageRela = relaTimesRele/sum(totReleDict["conts"][col])
  for i in range(len(opModel["conts"][col])):
   opModel["conts"][col][i][1] /= averageRela
  opModel["BASE_VALUE"] *= averageRela
 
 for col in totReleDict["cats"]:
  relaTimesRele = 0
  skeys = get_sorted_keys(model['cats'][col])
  for i in range(len(skeys)):
   relaTimesRele += opModel["cats"][col]["uniques"][skeys[i]] * totReleDict["cats"][col][i]
  relaTimesRele += opModel["cats"][col]["OTHER"] * totReleDict["cats"][col][-1]
  averageRela = relaTimesRele/sum(totReleDict["cats"][col])
  for i in range(len(skeys)):
   opModel["cats"][col]["uniques"][skeys[i]] /= averageRela
  opModel["cats"][col]["OTHER"] /= averageRela
  opModel["BASE_VALUE"] *= averageRela
 
 return opModel



def enforce_min_rela(model, minRela=0.1): #Not expanded to interxes 
 
 opModel = copy.deepcopy(model)
 
 for col in opModel["conts"]:
  for i in range(len(opModel["conts"][col])):
   opModel["conts"][col][i][1] = max(minRela, opModel["conts"][col][i][1])
 
 for col in opModel["cats"]:
  for u in opModel["cats"][col]["uniques"]:
   opModel["cats"][col]["uniques"][u] = max(minRela, opModel["cats"][col]["uniques"][u])
  opModel["cats"][col]["OTHER"] = max(minRela, opModel["cats"][col]["OTHER"])
 
 return opModel


def caricature_this_cont_col(model, col, mult=1,frac=1,defaultValue=1):
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for i in range(len(opModel["conts"][col])):
  opModel["conts"][col][i][1] = defaultValue + mult*(opModel["conts"][col][i][1]-defaultValue)
 
 return opModel


def caricature_this_cat_col(model, col, mult=1,frac=1,defaultValue=1):
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for u in opModel["cats"][col]["uniques"]:
  opModel["cats"][col]["uniques"][u] = defaultValue + mult*(opModel["cats"][col]["uniques"][u]-defaultValue)
 
 opModel["cats"][col]["OTHER"] = defaultValue + mult*(opModel["cats"][col]["OTHER"]-defaultValue)
 
 return opModel


def caricature_model(model, mult=1, frac=0.5, defaultValue=1): #Not expanded to interxes
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for col in opModel["conts"]:
  opModel = caricature_this_cont_col(opModel, col, mult, 1, defaultValue)
 
 for col in opModel["cats"]:
  opModel = caricature_this_cat_col(opModel, col, mult, 1, defaultValue)
 
 return opModel

#Functions for evaluating and auditioning.

def get_importance_of_this_cont_col(df, model, col, defaultValue=1):
 effects = get_effect_of_this_cont_col(df, model, col)
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_cat_col(df, model, col, defaultValue=1):
 effects = get_effect_of_this_cat_col(df, model, col)
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_contcont(df, model, cols, defaultValue=1):
 effects = get_effect_of_this_contcont(df, model, cols)
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_catcont(df, model, cols, defaultValue=1):
 effects = get_effect_of_this_catcont(df, model, cols)
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_catcat(df, model, cols, defaultValue=1):
 effects = get_effect_of_this_catcat(df, model, cols)
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def list_importances(df, model):
 op = {"col":[],"type":[],"imp":[]}
 if "cats" in model:
  for col in model["cats"]:
   op["col"].append(col)
   op["type"].append("cat")
   op['imp'].append(get_importance_of_this_cat_col(df, model, col))
 if "conts" in model:
  for col in model["conts"]:
   op["col"].append(col)
   op["type"].append("cont")
   op['imp'].append(get_importance_of_this_cont_col(df, model, col))
 if "catcats" in model:
  for cols in model["catcats"]:
   op["col"].append(cols)
   op["type"].append("catcat")
   op['imp'].append(get_importance_of_this_catcat(df, model, cols))
 if "catconts" in model:
  for cols in model["catconts"]:
   op["col"].append(cols)
   op["type"].append("catcont")
   op['imp'].append(get_importance_of_this_catcont(df, model, cols))
 if "contconts" in model:
  for cols in model["contconts"]:
   op["col"].append(cols)
   op["type"].append("contcont")
   op['imp'].append(get_importance_of_this_contcont(df, model, cols))
 op = pd.DataFrame(op)
 op = op.sort_values(["imp"], ascending=False).reset_index().drop("index", axis=1)
 return op

def suggest_interactions_based_on_importances(df, model):
 impDf = list_importances(df,model)
 count=0
 simpDf = impDf[impDf['type'].isin(["cont","cat"])].reset_index()
 op = {'interx':[],"type":[],'imp':[]}
 for i in range(len(simpDf)):
  for j in range(i+1, len(simpDf)):
   interx1 = simpDf["col"][i]+" X "+simpDf["col"][j]
   interx2 = simpDf["col"][i]+" X "+simpDf["col"][j]
   t = simpDf['type'][i]+simpDf['type'][j]
   if t=="contcat":
    t="catcont"
   imp = simpDf["imp"][i]*simpDf["imp"][j] #should this be multiplicative or additive?
   if (interx1 not in impDf["col"]) and (interx2 not in impDf["col"]):
    op['interx'].append(interx1)
    op['type'].append(t)
    op['imp'].append(imp)
 op = pd.DataFrame(op)
 op = op.sort_values(["imp"], ascending=False).reset_index().drop("index", axis=1)
 return op


def audition_this_cat(cat, inputSeries, err):
 assert(len(err)==len(inputSeries))
 rel = rele.produce_cat_relevances(inputSeries, cat)
 promises = np.matmul(err, rel)
 return sum(abs(promises))/len(err)

def audition_this_cont(cont, inputSeries, err):
 assert(len(err)==len(inputSeries))
 rel = rele.produce_cont_relevances(inputSeries, cont)
 promises = np.matmul(err, rel)
 return sum(abs(promises))/len(err)

def audition_this_catcat(catcat, inputSeries1, inputSeries2, err):
 assert(len(err)==len(inputSeries1))
 rel = rele.produce_catcat_relevances(inputSeries1, inputSeries2, catcat)
 promises = np.matmul(err, rel)
 print(promises)
 print(abs(promises))
 return sum(abs(promises))/len(err)
 
def audition_this_catcont(catcont, inputSeries1, inputSeries2, err):
 assert(len(err)==len(inputSeries1))
 rel = rele.produce_catcont_relevances(inputSeries1, inputSeries2, catcont)
 promises = np.matmul(err, rel)
 print(promises)
 print(abs(promises))
 return sum(abs(promises))/len(err)

def audition_this_contcont(contcont, inputSeries1, inputSeries2, err):
 assert(len(err)==len(inputSeries1))
 rel = rele.produce_contcont_relevances(inputSeries1, inputSeries2, contcont)
 promises = np.matmul(err, rel)
 print(promises)
 print(abs(promises))
 return sum(abs(promises))/len(err)


if __name__ == '__main__':
 exampleModel = {"BASE_VALUE": 1700, "cats":{'cat1':{'uniques':{'a':1,'b':1,'c':1,'d':1},'OTHER':1},'cat2':{'uniques':{'w':1,'x':1.1,'y':1,'z':1},'OTHER':1}}, "conts":{'cont1':[[0,1],[50,1.1]],'cont2':[[0,0.5],[20,1.5]]}}
 
 exampleDf = pd.DataFrame({"cont1":[0,25,50, 30], "cont2":[2,3,8,19], "cat1":["a","b","c","d"],"cat2":["w",'x','y','z'], "y":[5,7,9,11]})
 
 print(get_importance_of_this_cont_col(exampleDf, exampleModel, "cont1"))
 print(get_importance_of_this_cont_col(exampleDf, exampleModel, "cont2"))
 print(get_importance_of_this_cat_col(exampleDf, exampleModel, "cat1"))
 print(get_importance_of_this_cat_col(exampleDf, exampleModel, "cat2"))
 
 print(list_importances(exampleDf,exampleModel))
 print(suggest_interactions_based_on_importances(exampleDf,exampleModel))
 

if False: #__name__ == '__main__':
 exampleModel = {"BASE_VALUE": 1700, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}, 'b':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}, 'c':{'uniques':{'x':1.1,'y':1.2,'z':1.7},'OTHER':1}}, 'OTHER':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}}}, 'catconts':{'cat1 X cont1':{'uniques':{'a':[[0,0.8], [50,1.2]], 'b':[[0,0.8], [50,1.2]], 'c':[[0,0.8], [50,1.8]]}, 'OTHER':[[0,0.8], [50,1.2]]}}, 'contconts':{'cont1 X cont2': [[0,[[0,1.1], [20,1.2]]], [50,[[0,1.3], [20,1.4]]]]}}
 exampleDf = pd.DataFrame({"cont1":[0,25,50, 30], "cont2":[2,3,8,19], "cat1":["a","b","c","d"],"cat2":["w",'x','y','z'], "y":[5,7,9,11]})
 
 print(get_effect_of_this_catcat(exampleDf, exampleModel, "cat1 X cat2"))
 print(get_effect_of_this_catcont(exampleDf, exampleModel, "cat1 X cont1"))
 
 exampleDf = pd.DataFrame({"cont1":[25], "cont2":[10], 'y':[5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))
 
 exampleDf = pd.DataFrame({"cont1":[-999,999,-999,999], "cont2":[-999,-999,999,999], 'y':[5,5,5,5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))
 
 exampleDf = pd.DataFrame({"cont1":[-999,10,999,10], "cont2":[10,-999,10,999], 'y':[5,5,5,5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))

if False:#__name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}
 exampleDf = pd.DataFrame({"cont1":[0.013,0.015,0.025, 0.035], "cont2":[37,48,45,51], "cat1":["wstfgl","florpalorp","dukis","welp"], "y":[5,7,9,11]})
 
 print(get_effect_of_this_cont_col_on_single_input(0.012, exampleModel['conts']["cont1"])) #should be 1.02
 print(get_effect_of_this_cont_col_on_single_input(0.04, exampleModel['conts']["cont1"])) #should be 1.06
 print(get_effect_of_this_cat_col_on_single_input("florpalorp", exampleModel['conts']["cat1"])) #should be 0.92
 print(get_effect_of_this_cat_col_on_single_input(12, exampleModel['conts']["cat1"])) #should be 1.04
 
 print(list(get_effect_of_this_cat_col(exampleDf, exampleModel, "cat1"))) #[1.05,0.92,1.04,1.04]
 print(list(get_effect_of_this_cont_col(exampleDf, exampleModel, "cont1"))) #[1.03,1.05,1.08,1.06]

 print(caricature_model(exampleModel,2, 0.5))
 
 
if False: #__name__ == '__main__':
 df =  pd.DataFrame({"cat1":['a','a','a','a','b'],'cat2':['d','c','d','c','c'],"y":[1,1,1,1,2]})
 model = {"BASE_VALUE":1.2, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_cat({"OTHER":1,"uniques":{"a":1, "b":1}}, df['cat1'], err))
 print(audition_this_cat({"OTHER":1,"uniques":{"c":1, "d":1}}, df['cat2'], err))
 
 df =  pd.DataFrame({"cont1":[1,2,3,4,5],'cont2':[1,2,3,4,3],'cont3':[1,2,3,2,1],"y":[1,2,3,4,5]})
 model = {"BASE_VALUE":3, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_cont([[1,1],[5,1]], df['cont1'], err))
 print(audition_this_cont([[1,1],[4,1]], df['cont2'], err))
 print(audition_this_cont([[1,1],[3,1]], df['cont3'], err))
 
 df =  pd.DataFrame({"cat1":['a','a','a','a','b','b','b','b'], 'cat2':['c','c','d','d','c','c','d','d'], 'cat3':['e','f','e','f','e','f','e','f'], "y":[1,1,1,1,1,1,2,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"c":1, "d":1}},"uniques":{"a":{"OTHER":1,"uniques":{"c":1, "d":1}}, "b":{"OTHER":1,"uniques":{"c":1, "d":1}}}}, df['cat1'], df['cat2'], err))
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"e":1, "f":1}},"uniques":{"c":{"OTHER":1,"uniques":{"e":1, "f":1}}, "d":{"OTHER":1,"uniques":{"e":1, "f":1}}}}, df['cat2'], df['cat3'], err))
 print(audition_this_catcat({"OTHER":{"OTHER":1,"uniques":{"e":1, "f":1}},"uniques":{"a":{"OTHER":1,"uniques":{"e":1, "f":1}}, "b":{"OTHER":1,"uniques":{"e":1, "f":1}}}}, df['cat1'], df['cat3'], err))
 
 df =  pd.DataFrame({"cat1":['a','a','a','a','b','b','b','b'], 'cat2':['c','c','d','d','c','c','d','d'], 'cont1':[1,2,1,2,1,2,1,2], 'cont2':[1,1000,1,1000,1,1000,1,1000], "y":[1,1,1,1,1,2,1,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"a":[[1,1],[2,1]], "b":[[1,1],[2,1]]}}, df['cat1'], df['cont1'], err))
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"c":[[1,1],[2,1]], "d":[[1,1],[2,1]]}}, df['cat2'], df['cont1'], err))
 
 print(audition_this_catcont({"OTHER":[[1,1],[2,1]],"uniques":{"a":[[1,1],[2,1]], "b":[[1,1],[2,1]]}}, df['cat1'], df['cont2'], err))
 
 
 
 df =  pd.DataFrame({"cont1":[1,1,1,1,2,2,2,2], 'cont2':[1,1,2,2,1,1,2,2], 'cont3':[1,2,1,2,1,2,1,2], "y":[1,1,1,1,1,1,2,2]})
 model = {"BASE_VALUE":1.25, 'conts':{}, 'cats':{}}
 err = misc.predict(df, model) - df["y"]
 print(err)
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont1'], df['cont2'], err))
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont2'], df['cont3'], err))
 print(audition_this_contcont([[1,[[1,1],[2,1]]],[2,[[1,1],[2,1]]]], df['cont1'], df['cont3'], err))


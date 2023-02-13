import pandas as pd
import numpy as np
import math
import copy
import time
import datetime
import json
import os

from . import util
from . import calculus
from . import rele
from . import prep


def save_model(model, name="model", folder="models", timing=True):
 try:
  os.mkdir(folder)
 except:
  pass
 
 fullName = name
 if timing:
  now = datetime.datetime.now()
  fullName = fullName + "-" + str(now.year) + "-" + str(now.month) + "-" + str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)+ "-" + str(now.second)  
 
 with open(folder+'/'+fullName+".txt",'w') as f:
  f.write(json.dumps(model))
 #with open(fullName+".txt",'w') as f:


def load_model(filename):
 file = open(filename)
 return json.load(file)


def ratio_to_frac(x):
 return x/(x+1)

def frac_to_ratio(x):
 return 1/((1/x)-1)


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
 skeys = util.get_sorted_keys(model['cats'][col]["uniques"])
 postmultmat = np.array([model["cats"][col]["uniques"][key] for key in skeys]+[model["cats"][col]["OTHER"]])
 return np.matmul(reles,postmultmat)
 
def get_effects_of_cat_cols_from_relevance_dict(releDict, model):
 opDict = {}
 if "cats" in model:
  for col in model["cats"]:
   opDict[col]= get_effect_of_this_cat_col_from_relevances(releDict[col], model, col)
 return opDict

def get_effect_of_this_catcat_from_relevances(reles, model, cols):
 skeys1 = util.get_sorted_keys(model['catcats'][cols]["uniques"])
 skeys2 = util.get_sorted_keys(model['catcats'][cols]["OTHER"]["uniques"])
 postmultmat = []
 for key1 in skeys1:
  postmultmat = postmultmat+[model['catcats'][cols]['uniques'][key1]['uniques'][key2] for key2 in skeys2]+[model['catcats'][cols]['uniques'][key1]['OTHER']]
 postmultmat = postmultmat + [model['catcats'][cols]['OTHER']['uniques'][key2] for key2 in skeys2]+ [model['catcats'][cols]['OTHER']['OTHER']]
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effect_of_this_catcont_from_relevances(reles, model, cols, defaultValue=1):
 skeys = util.get_sorted_keys(model["catconts"][cols]["uniques"])
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

def get_effect_of_this_cont_col(ser, cont):
 x = ser
 effectOfCol = pd.Series([1]*len(ser))
 effectOfCol.loc[(x<=cont[0][0])] = cont[0][1] #Everything too early gets with the program
 for i in range(len(cont)-1):
  x1 = cont[i][0]
  x2 = cont[i+1][0]
  y1 = cont[i][1]
  y2 = cont[i+1][1]
  effectOfCol.loc[(x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[x>=cont[-1][0]] = cont[-1][1] #Everything too late gets with the program
 return effectOfCol

def get_effect_of_this_cat_on_single_input(x, cat): #slightly roundabout approach so we can copy for columns
 for unique in cat["uniques"]:
  if x==unique:
   return cat["uniques"][unique]
 return cat["OTHER"]

def get_effect_of_this_cat_col(ser, cat):
 effectOfCol = pd.Series([cat["OTHER"]]*len(ser))
 for unique in cat["uniques"]:
  effectOfCol[ser==unique] = cat["uniques"][unique]
 return effectOfCol

def get_effect_of_this_catcat_on_single_input(x1, x2, catcat):
 for unique in catcat["uniques"]:
  if x1==unique:
   return get_effect_of_this_cat_on_single_input(x2, catcat["uniques"][unique])
 return get_effect_of_this_cat_on_single_input(x2, catcat["OTHER"])

def get_effect_of_this_catcat(ser1, ser2, catcat):
 effectOfCol = pd.Series([catcat["OTHER"]["OTHER"]]*len(ser1))
 for unique1 in catcat['uniques']:
  for unique2 in catcat['uniques'][unique1]['uniques']:
   effectOfCol[(ser1==unique1) & (ser2==unique2)] = catcat['uniques'][unique1]['uniques'][unique2]
  effectOfCol[(ser1==unique1) & (~ser2.isin(catcat['uniques'][unique1]['uniques']))] = catcat['uniques'][unique1]['OTHER']
 for unique2 in catcat['OTHER']['uniques']:
  effectOfCol[(~ser1.isin(catcat['uniques'])) & (ser2==unique2)] = catcat['OTHER']['uniques'][unique2]
 return effectOfCol

def get_effect_of_this_catcont_on_single_input(x1, x2, catcont):
 for unique in catcont["uniques"]:
  if x1==unique:
   return get_effect_of_this_cont_on_single_input(x2, catcont["uniques"][unique])
 return get_effect_of_this_cont_on_single_input(x2, catcont["OTHER"])

def get_effect_of_this_catcont(ser1, ser2, catcont):
 
 x = ser2
 effectOfCol = pd.Series([1]*len(ser1))
 
 for unique in catcont['uniques']:
  effectOfCol.loc[(ser1==unique) & (x<=catcont['uniques'][unique][0][0])] = catcont['uniques'][unique][0][1] #Everything too early gets with the program
  for i in range(len(catcont['uniques'][unique])-1):
   x1 = catcont['uniques'][unique][i][0]
   x2 = catcont['uniques'][unique][i+1][0]
   y1 = catcont['uniques'][unique][i][1]
   y2 = catcont['uniques'][unique][i+1][1]
   effectOfCol.loc[(ser1==unique) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
  effectOfCol.loc[(ser1==unique) & (x>=catcont['uniques'][unique][-1][0])] = catcont['uniques'][unique][-1][1] #Everything too late gets with the program
  
 effectOfCol.loc[(~ser1.isin(catcont['uniques'])) & (x<=catcont['OTHER'][0][0])] = catcont['OTHER'][0][1] #Everything too early gets with the program
 for i in range(len(catcont['OTHER'])-1):
  x1 = catcont['OTHER'][i][0]
  x2 = catcont['OTHER'][i+1][0]
  y1 = catcont['OTHER'][i][1]
  y2 = catcont['OTHER'][i+1][1]
  effectOfCol.loc[(~ser1.isin(catcont['uniques'])) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[(~ser1.isin(catcont['uniques'])) & (x>=catcont['OTHER'][-1][0])] = catcont['OTHER'][-1][1] #Everything too late gets with the program
 
 return effectOfCol

#this next one is ludicrously inefficient but if you're using it in a context where that matters you made some bad life choices somewhere
def get_effect_of_this_contcont_on_single_input(x1,x2, contcont):
 x1cont = [[c[0], get_effect_of_this_cont_on_single_input(x2, c[1])] for c in contcont]
 return get_effect_of_this_cont_on_single_input(x1, x1cont)




def get_effect_of_this_contcont(ser1, ser2, contcont): #we are using x and y to predict z
 
 x = ser1
 y = ser2
 effectOfCol = pd.Series([1]*len(ser1))
 
 #Corners get with the program
 
 effectOfCol.loc[(x<=contcont[0][0]) & (y<=contcont[0][1][0][0])] = contcont[0][1][0][1]
 effectOfCol.loc[(x>=contcont[-1][0]) & (y<=contcont[-1][1][0][0])] = contcont[-1][1][0][1]
 effectOfCol.loc[(x<=contcont[0][0]) & (y>=contcont[0][1][-1][0])] = contcont[0][1][-1][1]
 effectOfCol.loc[(x>=contcont[-1][0]) & (y>=contcont[-1][1][-1][0])] = contcont[-1][1][-1][1]
 
 #Edges get with the program
 
 for i in range(len(contcont)-1):
  x1 = contcont[i][0]
  x2 = contcont[i+1][0]
  z1 = contcont[i][1][0][1]
  z2 = contcont[i+1][1][0][1]
  effectOfCol.loc[(y<=contcont[i][1][0][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(contcont)-1):
  x1 = contcont[i][0]
  x2 = contcont[i+1][0]
  z1 = contcont[i][1][-1][1]
  z2 = contcont[i+1][1][-1][1]
  effectOfCol.loc[(y>=contcont[i][1][-1][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(contcont[0][1])-1):
  y1 = contcont[0][1][i][0]
  y2 = contcont[0][1][i+1][0]
  z1 = contcont[0][1][i][1]
  z2 = contcont[0][1][i+1][1]
  effectOfCol.loc[(x<=contcont[0][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 for i in range(len(contcont[-1][1])-1):
  y1 = contcont[-1][1][i][0]
  y2 = contcont[-1][1][i+1][0]
  z1 = contcont[-1][1][i][1]
  z2 = contcont[-1][1][i+1][1]
  effectOfCol.loc[(x>=contcont[-1][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 #The interior
 
 for i in range(len(contcont)-1):
  x1 = contcont[i][0]
  x2 = contcont[i+1][0]
  for j in range(len(contcont[i][1])-1):
   y1 = contcont[0][1][j][0]
   y2 = contcont[0][1][j+1][0]
   z11 = contcont[i][1][j][1]
   z12 = contcont[i][1][j+1][1]
   z21 = contcont[i+1][1][j][1]
   z22 = contcont[i+1][1][j+1][1]
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
   effectOfCol = get_effect_of_this_cont_col(inputDf[col], model['conts'][col])
   preds = preds*effectOfCol
 if "cats" in model:
  for col in model["cats"]:
   effectOfCol = get_effect_of_this_cat_col(inputDf[col], model['cats'][col])
   preds = preds*effectOfCol
 if "catcats" in model:
  for cols in model["catcats"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_catcat(inputDf[col1], inputDf[col2], model['catcats'][cols])
   preds = preds*effectOfCol
 if "catconts" in model:
  for cols in model["catconts"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_catcont(inputDf[col1], inputDf[col2], model['catconts'][cols])
   preds = preds*effectOfCol
 if "contconts" in model:
  for cols in model["contconts"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_contcont(inputDf[col1], inputDf[col2], model['contconts'][cols])
   preds = preds*effectOfCol
 if "flink" in model:
  effectOfFlink = get_effect_of_this_cont_col(preds, model["flink"])
  preds = preds*effectOfFlink
 
 return preds

def predict_addl(inputDf, model):
 preds = pd.Series([model["BASE_VALUE"]]*len(inputDf))
 if "conts" in model:
  for col in model["conts"]:
   effectOfCol = get_effect_of_this_cont_col(inputDf[col], model['conts'][col])
   preds = preds+effectOfCol
 if "cats" in model:
  for col in model["cats"]:
   effectOfCol = get_effect_of_this_cat_col(inputDf[col], model['cats'][col])
   preds = preds+effectOfCol
 if "catcats" in model:
  for cols in model["catcats"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_catcat(inputDf[col1], inputDf[col2], model['catcats'][cols])
   preds = preds+effectOfCol
 if "catconts" in model:
  for cols in model["catconts"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_catcont(inputDf[col1], inputDf[col2], model['catconts'][cols])
   preds = preds+effectOfCol
 if "contconts" in model:
  for cols in model["contconts"]:
   col1, col2 = cols.split(' X ')
   effectOfCol = get_effect_of_this_contcont(inputDf[col1], inputDf[col2], model['contconts'][cols])
   preds = preds+effectOfCol
 if "flink" in model:
  effectOfFlink = get_effect_of_this_cont_col(preds, model["flink"])
  preds = preds+effectOfFlink
 return preds

def predict_model(inputDf, model, linkage = "Unity"):
 
 df = inputDf.reset_index(drop=False)
 
 if "featcomb" in model:
  if model["featcomb"]=="addl":
   comb = predict_addl(df, model)
  else:
   comb = predict_mult(df, model)
 else:
  comb = predict_mult(df, model)
 
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
 print(model)
 print("BASE_VALUE", round_to_sf(model["BASE_VALUE"], sf))
 if "flink" in model:
  print("flink", roundify_cont(model["flink"], sf))
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
  skeys = util.get_sorted_keys(model['cats'][col]["uniques"])
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

def get_importance_of_this_cont_col(inputDf, model, col, defaultValue=1):
 
 df = inputDf.reset_index(drop=True)
 
 effects = get_effect_of_this_cont_col(df[col], model['conts'][col])
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_cat_col(inputDf, model, col, defaultValue=1):
 
 df = inputDf.reset_index(drop=True)
 
 effects = get_effect_of_this_cat_col(df[col], model['cats'][col])
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_contcont(inputDf, model, cols, defaultValue=1):
 
 df = inputDf.reset_index(drop=True)
 
 col1, col2 = cols.split(' X ')
 effects = get_effect_of_this_contcont(df[col1],df[col2], model['contconts'][cols])
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_catcont(inputDf, model, cols, defaultValue=1):
 
 df = inputDf.reset_index(drop=True)
 
 col1, col2 = cols.split(' X ')
 effects = get_effect_of_this_catcont(df[col1],df[col2], model['catconts'][cols])
 if defaultValue==1:
  effects = effects/effects.mean()
 if defaultValue==0:
  effects = effects-effects.mean()
 effects = (effects-defaultValue).apply(abs)
 return effects.mean()

def get_importance_of_this_catcat(inputDf, model, cols, defaultValue=1):
 
 df = inputDf.reset_index(drop=True)
 
 col1, col2 = cols.split(' X ')
 effects = get_effect_of_this_catcat(df[col1],df[col2], model['catcats'][cols])
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


if __name__ == '__main__':
 pass
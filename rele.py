import numpy as np
import pandas as pd
import util

def produce_cont_relevances(inputSeries, cont):
 reles=np.zeros((len(cont)+1,len(inputSeries)))
 
 reles[0][(inputSeries<=cont[0][0])] = 1 #d(featpred)/d(pt)
 for i in range(len(cont)-1):
  x = inputSeries
  x1 = cont[i][0]
  x2 = cont[i+1][0]
  subset = (x>=x1) & (x<=x2)
  reles[i][subset] = (x2 - x[subset])/(x2 - x1) #d(featpred)/d(pt)
  reles[i+1][subset] = (x[subset] - x1)/(x2 - x1) #d(featpred)/d(pt)
 reles[-2][(inputSeries>=cont[-1][0])] = 1 #d(featpred)/d(pt)
 
 reles[-1] = 1 - np.sum(reles, axis=0)
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cont_relevances_dict(inputDf, model):
 opDict = {}
 
 if "conts" in model:
  for col in model["conts"]:
   opDict[col]=produce_cont_relevances(inputDf[col], model['conts'][col])
 
 return opDict

def produce_cat_relevances(inputSeries, cat):
 reles=np.zeros((len(cat["uniques"])+1,len(inputSeries)))
 
 skeys = util.get_sorted_keys(cat["uniques"])
 for i in range(len(skeys)):
  reles[i][inputSeries.isin([skeys[i]])] = 1 #d(featpred)/d(pt)
 reles[-1][~inputSeries.isin(skeys)] = 1 #d(featpred)/d(pt)
 
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cat_relevances_dict(inputDf, model):
 opDict = {}
 
 if "cats" in model:
  for col in model["cats"]:
   opDict[col]=produce_cat_relevances(inputDf[col], model['cats'][col])
 
 return opDict

#Interactions

def interact_relevances(relesA, relesB):
 
 relesA = np.transpose(relesA) #Yes, I know.
 relesB = np.transpose(relesB) #Shut up.
 
 relesI = np.zeros((len(relesA)*len(relesB),len(relesA[0])))
 
 for i in range(len(relesA)):
  for j in range(len(relesB)):
   relesI[i*len(relesB)+j] = relesA[i]*relesB[j]
 
 return np.transpose(relesI) # . . .

def produce_catcat_relevances(inputSeries1, inputSeries2, catcat):
 return interact_relevances(produce_cat_relevances(inputSeries1, catcat), produce_cat_relevances(inputSeries2, catcat["OTHER"]))

def produce_catcont_relevances(inputSeries1, inputSeries2, catcont):
 return interact_relevances(produce_cat_relevances(inputSeries1, catcont), produce_cont_relevances(inputSeries2, catcont["OTHER"]))

def produce_contcont_relevances(inputSeries1, inputSeries2, contcont):
 return interact_relevances(produce_cont_relevances(inputSeries1, contcont), produce_cont_relevances(inputSeries2, contcont[0][1]))

def produce_interxn_relevances_dict(inputDf, model):
 
 opDict = {}
 if 'catcats' in model:
  for cols in model['catcats']:
   col1, col2 = cols.split(' X ')
   opDict[cols] = produce_catcat_relevances(inputDf[col1], inputDf[col2], model['catcats'][cols])
 
 if 'catconts' in model:
  for cols in model['catconts']:
   col1, col2 = cols.split(' X ')
   opDict[cols] = produce_catcont_relevances(inputDf[col1], inputDf[col2], model['catconts'][cols])
 
 if 'contconts' in model:
  for cols in model['contconts']:
   col1, col2 = cols.split(' X ')
   opDict[cols] = produce_contcont_relevances(inputDf[col1], inputDf[col2], model['contconts'][cols])
 
 return opDict






def sum_and_listify_matrix(a):
 return np.array(sum(a)).tolist()

def produce_total_irelevances_dict(releDict):
 op = {}
 for cols in releDict:
  op[cols] = sum_and_listify_matrix(releDict[cols])
 return op

def produce_total_relevances_dict(contReleDict, catReleDict):
 op = {"conts":{},"cats":{}}
 for col in contReleDict:
  op["conts"][col] = sum_and_listify_matrix(contReleDict[col])
 for col in catReleDict:
  op["cats"][col] = sum_and_listify_matrix(catReleDict[col])
 return op



def produce_wReleDict(releDict, w):
 wReleDict = {}
 for col in releDict:
  wReleDict[col]=w*releDict[col]
 return wReleDict
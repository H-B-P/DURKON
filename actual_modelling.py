import pandas as pd
import numpy as np
import math
import copy
import time

import util
import misc
import calculus
import pena
import rele
import prep

#pens, lrs, staticFeatLists, specificPenses and models are per model
#relelistlists are listed per df, then per model
#link, linkgrad are per pred; linkgrad is then per model
#lossgrad is per df, then per pred

def train_models(inputDfs, targets, nrounds, lrs, startingModels, weightCol=None, staticFeats = [], lras=None, lossgrads=[[calculus.Gauss_grad]], links=[calculus.Unity_link], linkgrads=[[calculus.Unity_link_grad]], pens=None, minRela=None, prints="verbose", record=False):
 
 history=[]
 
 inputDfs = [df.reset_index() for df in inputDfs]
 
 if type(targets)!=type([1,2,3]):
  targets=[targets]
 
 if lras==None:
  lras = [calculus.default_LRA]*len(startingModels)
 
 models = copy.deepcopy(startingModels)
 
 if prints!="silent":
  print("initial weights and relevances setup")
 
 contReleDictListList = []
 catReleDictListList = []
 totReleDictListList = []
 
 contWReleDictListList = []
 catWReleDictListList = []
 totWReleDictListList = []
 
 interReleDictListList = []
 totInterReleDictListList = []
 
 interWReleDictListList = []
 totInterWReleDictListList = []
 
 for inputDf in inputDfs:
  if weightCol==None:
   weight = np.ones(len(inputDf))
  else:
   weight = inputDf[weightCol]
  w = np.array(np.transpose(np.matrix(weight)))
  
  contReleDictList = []
  catReleDictList = []
  totReleDictList = []
  
  contWReleDictList = []
  catWReleDictList = []
  totWReleDictList = []
  
  interReleDictList = []
  totInterReleDictList = []
  
  interWReleDictList = []
  totInterWReleDictList = []
  
  for model in models:
   
   cord = rele.produce_cont_relevances_dict(inputDf,model) #d(feat)/d(pt)
   card = rele.produce_cat_relevances_dict(inputDf,model) #d(feat)/d(pt)
   tord = rele.produce_total_relevances_dict(cord, card)
   
   cowrd = rele.produce_wReleDict(cord, w) #d(feat)/d(pt), adjusted for weighting
   cawrd = rele.produce_wReleDict(card, w) #d(feat)/d(pt), adjusted for weighting
   towrd = rele.produce_total_relevances_dict(cowrd, cawrd)
   
   #Interactions . . .
   
   ird = rele.produce_interxn_relevances_dict(inputDf, model) #d(feat)/d(pt)
   tird = rele.produce_total_irelevances_dict(ird)
   
   wird = rele.produce_wReleDict(ird, w) #d(feat)/d(pt), adjusted for weighting
   twird = rele.produce_total_irelevances_dict(wird)
   
   contReleDictList.append(copy.deepcopy(cord))
   catReleDictList.append(copy.deepcopy(card))
   totReleDictList.append(copy.deepcopy(tord))
   
   contWReleDictList.append(copy.deepcopy(cowrd))
   catWReleDictList.append(copy.deepcopy(cawrd))
   totWReleDictList.append(copy.deepcopy(towrd))
   
   interReleDictList.append(copy.deepcopy(ird))
   totInterReleDictList.append(copy.deepcopy(tird))
   
   interWReleDictList.append(copy.deepcopy(wird))
   totInterWReleDictList.append(copy.deepcopy(twird))
  
  contReleDictListList.append(copy.deepcopy(contReleDictList))
  catReleDictListList.append(copy.deepcopy(catReleDictList))
  totReleDictListList.append(copy.deepcopy(totReleDictList))
  
  contWReleDictListList.append(copy.deepcopy(contWReleDictList))
  catWReleDictListList.append(copy.deepcopy(catWReleDictList))
  totWReleDictListList.append(copy.deepcopy(totWReleDictList))
  
  interReleDictListList.append(copy.deepcopy(interReleDictList))
  totInterReleDictListList.append(copy.deepcopy(totInterReleDictList))
  
  interWReleDictListList.append(copy.deepcopy(interWReleDictList))
  totInterWReleDictListList.append(copy.deepcopy(totInterWReleDictList))
  
  
 for i in range(nrounds):
  
  if prints!="silent":
   print("epoch: "+str(i+1)+"/"+str(nrounds))
  if prints=="verbose":
   for model in models:
    misc.explain(model)
  
  oModels=copy.deepcopy(models) #Duplicate so effects for all dfs are applied simultaneously
  
  for d in range(len(inputDfs)):
   
   inputDf = inputDfs[d]
   
   if (prints=="verbose") and (len(inputDfs)>1):
    print("adjusting to df #"+ str(d))
   
   if prints=="verbose":
    print("initial pred and effect-gathering")
   
   contEffectsList = []
   catEffectsList = []
   interxEffectsList = []
   
   combs = []
   
   for m in range(len(models)):
    
    oModel = oModels[m]
    
    contEffects = misc.get_effects_of_cont_cols_from_relevance_dict(contReleDictListList[d][m],oModel)
    catEffects = misc.get_effects_of_cat_cols_from_relevance_dict(catReleDictListList[d][m],oModel)
    interxEffects = misc.get_effects_of_interxns_from_relevance_dict(interReleDictListList[d][m],oModel)
    
    contEffectsList.append(contEffects)
    catEffectsList.append(catEffects)
    interxEffectsList.append(interxEffects)
    
    if model['featcomb']=="addl":
     comb = misc.comb_from_effects_addl(oModel["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
    else:
     comb = misc.comb_from_effects_mult(oModel["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
    
    combs.append(comb)
   
   #linkgradients = [[misc.lambdapply(linkgrad, combs) for linkgrad in linkgradset] for linkgradset in linkgrads]
   #preds = [misc.lambdapply(link, combs) for link in links]
   #lossgradients = [misc.lambdapply(lossgrad, preds+[df[target]]) for lossgrad in lossgrads[d]]
   #
   #print("linkg", linkgradients)
   #print('p', preds)
   #print("lossg", lossgradients)
   
   linkgradients = [[linkgrad(*combs) for linkgrad in linkgradset] for linkgradset in linkgrads]
   preds = [link(*combs) for link in links]
   tCols = [inputDf[target] for target in targets]
   lossgradients = [lossgrad(*preds,*tCols) for lossgrad in lossgrads[d]]
   
   #print(lossgradients)
   
   for p in range(len(preds)):
    
    for m in range(len(models)):
     
     model = models[m]
     
     lra = lras[m](*combs)
     
     cowrd = contWReleDictListList[d][m]
     cawrd = catWReleDictListList[d][m]
     wird = interWReleDictListList[d][m]
     
     linkgradient = linkgradients[p][m]
     lossgradient = lossgradients[p]
     
     if type(linkgradient)==type(pd.Series([])):
      
      if "conts" in model:
       if prints=="verbose":
        print("adjust conts")
       
       for col in [c for c in model['conts'] if c not in staticFeats]:
        
        if model['featcomb']=="addl":
         finalGradients = np.matmul(np.array(lossgradient*linkgradient),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
        else:
         effectOfCol = contEffectsList[m][col]
         ceoc = combs[m]/effectOfCol #d(comb)/d(feat)
         finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
        
        for k in range(len(model['conts'][col])):
         totRele = sum([totWReleDictListList[D][m]["conts"][col][k] for D in range(len(inputDfs))])
         if totRele>0:
          model["conts"][col][k][1] -= finalGradients[k]*lrs[m]*lra/totRele
      
      if "cats" in model:
       if prints=="verbose":
        print("adjust cats")
       
       for col in [c for c in model['cats'] if c not in staticFeats]:
        
        if model['featcomb']=="addl":
         finalGradients = np.matmul(np.array(lossgradient*linkgradient),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
        else:
         effectOfCol = catEffectsList[m][col]
         ceoc = combs[m]/effectOfCol #d(comb)/d(feat)
         finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
        
        skeys = util.get_sorted_keys(model['cats'][col]["uniques"])
        
        #all the uniques . . .
        for k in range(len(skeys)):
         totRele = sum([totWReleDictListList[D][m]["cats"][col][k] for D in range(len(inputDfs))])
         if totRele>0:
          model["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrs[m]*lra/totRele
         
        # . . . and "OTHER"
        totRele = towrd["cats"][col][-1]
        if totRele>0:
         model["cats"][col]["OTHER"] -= finalGradients[-1]*lrs[m]*lra/totRele
        
      if "catcats" in model:
       if prints=="verbose":
        print('adjust catcats')
       
       for cols in [c for c in model['catcats'] if c not in staticFeats]:
        
        if model['featcomb']=="addl":
         finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
        else:
         effectOfCols = interxEffectsList[m][cols]
         ceoc = combs[m]/effectOfCols #d(comb)/d(feat)
         finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
        
        skeys1 = util.get_sorted_keys(model['catcats'][cols]["uniques"])
        skeys2 = util.get_sorted_keys(model['catcats'][cols]["OTHER"]["uniques"])
        
        for i in range(len(skeys1)):
         for j in range(len(skeys2)):
          totRele = sum([totInterWReleDictListList[D][m][cols][i*(len(skeys2)+1)+j] for D in range(len(inputDfs))])
          if totRele>0:
           model["catcats"][cols]["uniques"][skeys1[i]]['uniques'][skeys2[j]] -= finalGradients[i*(len(skeys2)+1)+j]*lrs[m]*lra/totRele
         totRele = sum([totInterWReleDictListList[D][m][cols][i*(len(skeys2)+1)+len(skeys2)] for D in range(len(inputDfs))])
         if totRele>0:
          model['catcats'][cols]["uniques"][skeys1[i]]['OTHER'] -= finalGradients[i*(len(skeys2)+1)+len(skeys2)]*lrs[m]*lra/totRele
        
        for j in range(len(skeys2)):
         totRele = sum([totInterWReleDictListList[D][m][cols][len(skeys1)*(len(skeys2)+1)+j] for D in range(len(inputDfs))])
         if totRele>0:
          model["catcats"][cols]['OTHER']['uniques'][skeys2[j]] -= finalGradients[len(skeys1)*(len(skeys2)+1)+j]*lrs[m]*lra/totRele
        
        totRele = sum([ totInterWReleDictListList[D][m][cols][-1] for D in range(len(inputDfs))])
        if totRele>0:
         model['catcats'][cols]['OTHER']['OTHER'] -= finalGradients[-1]*lrs[m]*lra/totRele
      
      if "catconts" in model:
       if prints=="verbose":
        print('adjust catconts')
       
       for cols in [c for c in model['catconts'] if c not in staticFeats]:
        
        if model['featcomb']=="addl":
         finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
        else:
         effectOfCols = interxEffectsList[m][cols]
         ceoc = combs[m]/effectOfCols #d(comb)/d(feat)
         finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
        
        skeys = util.get_sorted_keys(model['catconts'][cols]["uniques"])
        
        for i in range(len(skeys)):
         for j in range(len(model['catconts'][cols]["OTHER"])):
          totRele = sum([totInterWReleDictListList[D][m][cols][i*(len(model['catconts'][cols]["OTHER"])+1)+j] for D in range(len(inputDfs))])
          if totRele>0:
           model['catconts'][cols]['uniques'][skeys[i]][j][1] -= finalGradients[i*(len(model['catconts'][cols]["OTHER"])+1)+j]*lrs[m]*lra/totRele
        
        for j in range(len(model['catconts'][cols]["OTHER"])):
         totRele = sum([totInterWReleDictListList[D][m][cols][len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j] for D in range(len(inputDfs))])
         if totRele>0:
          model['catconts'][cols]['OTHER'][j][1] -= finalGradients[len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]*lrs[m]*lra/totRele
      
      if "contconts" in model:
       if prints=="verbose":
        print('adjust contconts')
       
       for cols in [c for c in model['contconts'] if c not in staticFeats]:
        
        if model['featcomb']=="addl":
         finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
        else:
         effectOfCols = interxEffectsList[m][cols]
         ceoc = combs[m]/effectOfCols #d(comb)/d(feat)
         finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
        
        for i in range(len(model['contconts'][cols])):
         for j in range(len(model['contconts'][cols][0][1])):
          totRele = sum([totInterWReleDictListList[D][m][cols][i*(len(model['contconts'][cols][0][1])+1)+j] for D in range(len(inputDfs))])
          if totRele>0:
           model['contconts'][cols][i][1][j][1] -= finalGradients[i*(len(model['contconts'][cols][0][1])+1)+j]*lrs[m]*lra/totRele
  
  #Penalize!
  if pens!=None:
   if prints=="verbose":
    print("penalties")
   for m in range(len(models)):
    if models[m]['featcomb']=="addl":
     models[m] = pena.penalize_model(models[m], pens[m]*lrs[m], 0)
    else:
     models[m] = pena.penalize_model(models[m], pens[m]*lrs[m], 1)
  
  if minRela!=None:
   models = [misc.enforce_min_rela(model, minRela) for model in models]
  
  history.append(models)
 
 if record: 
  return models, history
 return models

def train_model(inputDf, target, nrounds, lr, startingModel, weight=None, staticFeats = [], pen=0, specificPens={}, lossgrad=calculus.Poisson_grad, link=calculus.Unity_link, linkgrad=calculus.Unity_link_grad, minRela=None, prints="verbose", record=False):
 
 history=[]
 
 inputDf = inputDf.reset_index()
 
 model = copy.deepcopy(startingModel)
 
 if prints!="silent":
  print("initial weights and relevances setup")
 
 if weight==None:
  weight = np.ones(len(inputDf))
 w = np.array(np.transpose(np.matrix(weight)))
 
 cord = rele.produce_cont_relevances_dict(inputDf,model) #d(feat)/d(pt)
 card = rele.produce_cat_relevances_dict(inputDf,model) #d(feat)/d(pt)
 tord = rele.produce_total_relevances_dict(cord, card)
 
 cowrd = rele.produce_wReleDict(cord, w) #d(feat)/d(pt), adjusted for weighting
 cawrd = rele.produce_wReleDict(card, w) #d(feat)/d(pt), adjusted for weighting
 towrd = rele.produce_total_relevances_dict(cowrd, cawrd)
 
 #Interactions . . .
 
 ird = rele.produce_interxn_relevances_dict(inputDf, model) #d(feat)/d(pt)
 tird = rele.produce_total_irelevances_dict(ird)
 
 wird = rele.produce_wReleDict(ird, w) #d(feat)/d(pt), adjusted for weighting
 twird = rele.produce_total_irelevances_dict(wird)
 
 for i in range(nrounds):
  
  if prints!="silent":
   print("epoch: "+str(i+1)+"/"+str(nrounds))
  if prints=="verbose":
   misc.explain(model)
  
  if prints=="verbose":
   print("initial pred and effect-gathering")
  
  contEffects = misc.get_effects_of_cont_cols_from_relevance_dict(cord,model)
  catEffects = misc.get_effects_of_cat_cols_from_relevance_dict(card,model)
  interxEffects = misc.get_effects_of_interxns_from_relevance_dict(ird,model)
  
  if model['featcomb']=="addl":
   comb = misc.comb_from_effects_addl(model["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
  else:
   comb = misc.comb_from_effects_mult(model["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
  
  if "flink" in model:
   flinkEffect = misc.get_effect_of_this_cont_col(comb, model["flink"])
   preFlinkComb = comb
   if model['featcomb']=="addl":
    comb = comb+flinkEffect
   else:
    comb = comb*flinkEffect
  
  linkgradient = linkgrad(comb) #d(pred)/d(comb)
  
  pred = comb.apply(link)
  lossgradient = lossgrad(pred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  #Adjust adjustable model components one-by-one . . .
  
  if "flink" in model:
   if prints=="verbose":
    print("adjust flink")
   
   flinkRele = rele.produce_cont_relevances(preFlinkComb, model["flink"])#Has to be reproduced every turn because the dist changes over time (it better!)
   
   if model["featcomb"]=="addl":
    finalGradients = np.matmul(np.array(lossgradient*linkgradient),w*flinkRele)#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
   else:
    finalGradients = np.matmul(np.array(lossgradient*linkgradient*preFlinkComb),w*flinkRele) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
   
   
   totReles = rele.sum_and_listify_matrix(w*flinkRele)
   for k in range(len(model['flink'])):
    totRele = totReles[k]
    if totRele!=0:
     model["flink"][k][1] -= finalGradients[k]*lr/totRele
  
  if "conts" in model:
   if prints=="verbose":
    print("adjust conts")
   
   for col in [c for c in model['conts'] if c not in staticFeats]:
    
    if model['featcomb']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCol = contEffects[col]
     ceoc = comb/effectOfCol #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    for k in range(len(model['conts'][col])):
     totRele = towrd["conts"][col][k]
     if totRele>0:
      model["conts"][col][k][1] -= finalGradients[k]*lr/totRele
  
  if "cats" in model:
   if prints=="verbose":
    print("adjust cats")
   
   for col in [c for c in model['cats'] if c not in staticFeats]:
    
    if model['featcomb']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCol = catEffects[col]
     ceoc = comb/effectOfCol #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys = util.get_sorted_keys(model['cats'][col]["uniques"])
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = towrd["cats"][col][k]
     if totRele>0:
      model["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lr/totRele
     
    # . . . and "OTHER"
    totRele = towrd["cats"][col][-1]
    if totRele>0:
     model["cats"][col]["OTHER"] -= finalGradients[-1]*lr/totRele
    
  if "catcats" in model:
   if prints=="verbose":
    print('adjust catcats')
   
   for cols in [c for c in model['catcats'] if c not in staticFeats]:
    
    if model['featcomb']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys1 = util.get_sorted_keys(model['catcats'][cols]["uniques"])
    skeys2 = util.get_sorted_keys(model['catcats'][cols]["OTHER"]["uniques"])
    
    for i in range(len(skeys1)):
     for j in range(len(skeys2)):
      totRele = twird[cols][i*(len(skeys2)+1)+j]
      if totRele>0:
       model["catcats"][cols]["uniques"][skeys1[i]]['uniques'][skeys2[j]] -= finalGradients[i*(len(skeys2)+1)+j]*lr/totRele
     totRele = twird[cols][i*(len(skeys2)+1)+len(skeys2)]
     if totRele>0:
      model['catcats'][cols]["uniques"][skeys1[i]]['OTHER'] -= finalGradients[i*(len(skeys2)+1)+len(skeys2)]*lr/totRele
    
    for j in range(len(skeys2)):
     totRele = twird[cols][len(skeys1)*(len(skeys2)+1)+j]
     if totRele>0:
      model["catcats"][cols]['OTHER']['uniques'][skeys2[j]] -= finalGradients[len(skeys1)*(len(skeys2)+1)+j]*lr/totRele
     
    totRele = twird[cols][-1]
    if totRele>0:
     model['catcats'][cols]['OTHER']['OTHER'] -= finalGradients[-1]*lr/totRele
  
  if "catconts" in model:
   if prints=="verbose":
    print('adjust catconts')
  
   for cols in [c for c in model['catconts'] if c not in staticFeats]:
    
    if model['featcomb']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    skeys = util.get_sorted_keys(model['catconts'][cols]["uniques"])
    
    for i in range(len(skeys)):
     for j in range(len(model['catconts'][cols]["OTHER"])):
      totRele = twird[cols][i*(len(model['catconts'][cols]["OTHER"])+1)+j]
      if totRele>0:
       model['catconts'][cols]['uniques'][skeys[i]][j][1] -= finalGradients[i*(len(model['catconts'][cols]["OTHER"])+1)+j]*lr/totRele
    
    for j in range(len(model['catconts'][cols]["OTHER"])):
     totRele = twird[cols][len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]
     if totRele>0:
      model['catconts'][cols]['OTHER'][j][1] -= finalGradients[len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]*lr/totRele
  
  if "contconts" in model:
   if prints=="verbose":
    print('adjust contconts')
   
   for cols in [c for c in model['contconts'] if c not in staticFeats]:
    
    if model['featcomb']=="addl":
     finalGradients = np.matmul(np.array(lossgradient*linkgradient),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb) * d(comb)/d(pt)
    else:
     effectOfCols = interxEffects[cols]
     ceoc = comb/effectOfCols #d(comb)/d(feat)
     finalGradients = np.matmul(np.array(lossgradient*linkgradient*ceoc),wird[cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(comb)* d(comb)/d(feat) * d(feat)/d(pt)
    
    for i in range(len(model['contconts'][cols])):
     for j in range(len(model['contconts'][cols][0][1])):
      totRele = twird[cols][i*(len(model['contconts'][cols][0][1])+1)+j]
      if totRele>0:
       model['contconts'][cols][i][1][j][1] -= finalGradients[i*(len(model['contconts'][cols][0][1])+1)+j]*lr/totRele
  
  #Penalize!
  if pen>0:
   if prints=="verbose":
    print("penalties")
   if model['featcomb']=="addl":
    model = pena.penalize_model(model, pen*lr, 0, specificPens)
   else:
    model = pena.penalize_model(model, pen*lr, 1, specificPens)
   
   if minRela!=None:
    model = misc.enforce_min_rela(model, minRela)
 
  history.append(model)
 
 if record:
  return model, history
 return model


if __name__ == '__main__':
 df = pd.DataFrame({"cont1":[0,1,0,1],"cat1":[0,0,1,1],"y":[0,1,1,4]})
 model = {"BASE_VALUE":1.0,"conts":{"cont1":[[0,0],[1,1]]}, "cats":{"cat1":{"uniques":{0:0, 1:0},"OTHER":0}},'featcomb':'addl'}
 model = train_model(df, "y", 100, 0.1, model, lossgrad=calculus.Gauss_grad)
 print(model)
 print(misc.predict(df, model))
 df["preds"] = misc.predict(df, model)
 model["flink"] = prep.get_cont_feat(df, "preds",3, 0.001, 0)
 model = train_model(df, "y", 100, 0.1, model, lossgrad=calculus.Gauss_grad)
 print(model)
 print(misc.predict(df, model))





if False: #__name__ == '__main__':
 df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":{},'featcomb':'mult'}]
 newModels = train_models([df], "y",100, [0.02], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x1":[1,2,3,4],"x2":[1,2,3,2],"y":[1+1,2+2,3+3,4+2]})
 models = [{"BASE_VALUE":2,"conts":{"x1":[[1,1],[4,1]]}, "cats":{},'featcomb':'mult'}, {"BASE_VALUE":1,"conts":{"x2":[[1,1],[4,1]]}, "cats":{},'featcomb':'mult'}]
 newModels = train_models([df], "y",200, [0.1, 0.1], models, links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad, calculus.Add_mlink_grad]])
 print(newModels)
 for newModel in newModels:
  misc.explain(newModel)
 
 print(misc.predict_models(df, newModels, calculus.Add_mlink)) #It looks like it fails but it doesn't, put whatever base values or varying lrs you want and it'll appear to crash while predicting perfectly.
 
 df = pd.DataFrame({"x1":[1,2,3,4,1,2,3,4],"x2":[1,1,2,2,3,3,4,4],"y":[max(1,1),max(2,1),max(3,2),max(4,2),max(1,3),max(2,3),max(3,4),max(4,4)]})
 models = [{"BASE_VALUE":1,"conts":{"x1":[[1,1],[4,1]]}, "cats":{},'featcomb':'mult'}, {"BASE_VALUE":1,"conts":{"x2":[[1,1],[4,1]]}, "cats":{},'featcomb':'mult'}]
 newModels = train_models([df], "y",400, [0.1, 0.12], models, links=[calculus.Max_mlink_2], linkgrads=[[calculus.Max_mlink_grad_2_A, calculus.Max_mlink_grad_2_B]])
 print(newModels)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x1":[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],"x2":[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],"y":[1*1,2*1,3*1,4*1,1*2,2*2,3*2,4*2,1*3,2*3,3*3,4*3,1*4,2*4,3*4,4*4]})
 models = [{"BASE_VALUE":-3,"conts":{"x1":[[1,1],[4,1]]}, "cats":{},'featcomb':'addl'}, {"BASE_VALUE":-2,"conts":{"x2":[[1,1],[4,1]]}, "cats":{},'featcomb':'addl'}]
 newModels = train_models([df], "y",200, [0.01, 0.02], models, links=[calculus.Mult_mlink_2], linkgrads=[[calculus.Mult_mlink_grad_2_A, calculus.Mult_mlink_grad_2_B]]) 
 print(newModels)
 for newModel in newModels:
  misc.explain(newModel)
 
 print(misc.predict_models(df, newModels, calculus.Mult_mlink_2)) #It looks like it fails but it doesn't, put whatever base values or varying lrs you want and it'll appear to crash while predicting perfectly.
 
 df = pd.read_csv('gnormal.csv')
 modelU = {"BASE_VALUE":1.0,"conts":{"x":[[0,103], [7,103]]},'featcomb':'mult'}
 modelP = {"BASE_VALUE":1.0,"conts":{"x":[[0,0.2], [7,0.2]]},'featcomb':'mult'}
 cdf = df[df['censored']].reset_index()
 udf = df[~df['censored']].reset_index()
 
 print(len(cdf))
 print(len(udf))
 #assert(False)
 #newModels = train_models([udf], "y", 1000, [10,0.001], [modelU, modelP], lossgrads = [[calculus.gnormal_u_diff, calculus.gnormal_p_diff]], links=[calculus.Take0,calculus.Take1], linkgrads=[[calculus.js1, calculus.js0],[calculus.js0, calculus.js1]])
 
 newModels = train_models([udf, cdf], "y", 1000, [10,0.005], [modelU, modelP], lossgrads = [[calculus.gnormal_u_diff, calculus.gnormal_p_diff],[calculus.u_diff_censored,calculus.p_diff_censored]], links=[calculus.Take0,calculus.Take1], linkgrads=[[calculus.js1, calculus.js0],[calculus.js0, calculus.js1]])
 for newModel in newModels:
  misc.explain(newModel)









if False:#__name__ == '__main__':
 df = pd.DataFrame({"cat1":[True, True, True, False, False, False],'cat2':[True, False, True, False, True, False],"y":[2,1,2,0,1,0]})
 model = {"BASE_VALUE":1.0,"conts":{}, "featcomb":'addl', "cats":{"cat1":{"uniques":{True:1,False:1,},"OTHER":1}, "cat2":{"uniques":{True:1,False:1},"OTHER":1}}}
 model = train_model(df, 'y', 500, 0.1, model, lossgrad=calculus.Gauss_grad)
 print(model)
 


if False:#__name__ == '__main__':
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 print(rele.produce_catcat_relevances(df['cat1'], df['cat2'], model["catcats"]["cat1 X cat2"]))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':2},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 reles = rele.produce_catcat_relevances(df['cat1'], df['cat2'], model["catcats"]["cat1 X cat2"])
 print(misc.get_effect_of_this_catcat_from_relevances(reles, model, "cat1 X cat2"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,1,1,1,2,1,1,1,1,1]})
 model = {"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}
 
 newModel = train_model(df, "y",50, 0.4, model, staticFeats = ['cat1','cat2'])
 print(newModel)
 
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 print(rele.produce_catcont_relevances(df['cat1'], df['cont1'], model["catconts"]["cat1 X cont1"]))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,4],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 reles = rele.produce_catcont_relevances(df['cat1'], df['cont1'], model["catconts"]["cat1 X cont1"])
 print(misc.get_effect_of_this_catcont_from_relevances(reles, model, "cat1 X cont1"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1,1,1,2,1,1,1,1,1]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}
 newModel = train_model(df, "y",50, 0.4, model, staticFeats = ['cat1','cont1'])
 print(newModel)
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }
 print(rele.produce_contcont_relevances(df['cont1'], df['cont2'], model["contconts"]["cont1 X cont2"]))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,5]]]]} }]
 reles = rele.produce_contcont_relevances(df['cont1'], df['cont2'], model["contconts"]["cont1 X cont2"])
 print(misc.get_effect_of_this_contcont_from_relevances(reles, models[0], "cont1 X cont2"))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1, 1,1,1, 1,1,2, 1,1]})
 model = {"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }
 newModel = train_model(df, "y",50, 0.4, model, staticFeats = ['cont1','cont2'])
 print(newModel)
 
if False:
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[1,2,3,4,5,6,7,8,9]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,5],[5,5],[9,5]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2], models,weights=[1,2,3,4,5,6,7,8,9])
 for newModel in newModels:
  misc.explain(newModel)


if False:
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,1],[8,1]]}, "cats":[]}]
 print(produce_cont_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 print(produce_cat_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,2],[8,1]]}, "cats":[]}]
 reles = produce_cont_relevances(df, models[0], "x")
 print(misc.get_effect_of_this_cont_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":2,"mouse":1,"dog":3},"OTHER":1.5}}}]
 reles = produce_cat_relevances(df, models[0], "x")
 print(misc.get_effect_of_this_cat_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2],"y":[2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[2,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1, 995,996,997,998,999,1000],"y":[2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[1000,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [1.5], models)
 for newModel in newModels:
  misc.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3,4],"y":[1+1,2+1.2,3+1.4,4+1.6]})
 models = [{"BASE_VALUE":2,"conts":{"x":[[1,1],[4,1]]}, "cats":[]},{"BASE_VALUE":1,"conts":{"x":[[1,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [1.5, 1.5], models)
 for newModel in newModels:
  misc.explain(newModel)

 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2.0,3.0,2.0,1.0,1.5]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 newModels = train_model(df, "y",100, [0.5], models)
 for newModel in newModels:
  misc.explain(newModel)
 

import pandas as pd
import numpy as np
import math
import copy
import time

import util
import apply_model
import calculus


def construct_model(inputDf, target, nrounds, lr, startingModel, grad=calculus.Poisson_grad):
 
 model = copy.deepcopy(startingModel)
 
 for i in range(nrounds):
  
  print("epoch: "+str(i)+"/"+str(nrounds))
  apply_model.explain(model)
  
  pred = apply_model.predict(inputDf, model)
  
  gradient = grad(pred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  for col in model["conts"]:
   effectOfCol = apply_model.get_effect_of_this_cont_col(inputDf, model, col)
   
   peoc = pred/effectOfCol #d(pred)/d(featpred)
   
   totalAffectList=[0]*len(model["conts"][col])
   totalChangeList=[0]*len(model["conts"][col])
   
   #effect of outer range on starting point
   affectedness = pd.Series([0]*len(inputDf))
   affectedness.loc[(inputDf[col]<model["conts"][col][0][0])] = 1 #d(featpred)/d(pt)
   finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
   totalAffectList[0]+=sum(affectedness)
   totalChangeList[0]+=finalGradient*lr
   
   for i in range(len(model["conts"][col])-1):
    #define terms
    x = inputDf[col]
    x1 = model["conts"][col][i][0]
    x2 = model["conts"][col][i+1][0]
    #points are affected by what's ahead . . .
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(x>=x1) & (x<x2)] = (x2 - x)/(x2 - x1) #d(featpred)/d(pt)
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    totalAffectList[i]+=sum(affectedness)
    totalChangeList[i]+=finalGradient
    #. . . and what's behind.
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(x>=x1) & (x<x2)] = (x-x1)/(x2 - x1) #d(featpred)/d(pt)
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    totalAffectList[i+1]+=sum(affectedness)
    totalChangeList[i+1]+=finalGradient
  
   # effect of outer range on final point
   affectedness = pd.Series([0]*len(inputDf))
   affectedness.loc[(inputDf[col]>=model["conts"][col][-1][0])] = 1 #d(featpred)/d(pt)
   finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
   totalAffectList[i+1]+=sum(affectedness)
   totalChangeList[i+1]+=finalGradient
   
   for i in range(len(totalAffectList)):
    if totalAffectList[i]>0:
     model["conts"][col][i][1]-= lr*totalChangeList[i]/totalAffectList[i]
  
  for col in model["cats"]:
   
   effectOfCol = apply_model.get_effect_of_this_cont_col(inputDf, model, col)
   
   peoc = pred/effectOfCol #d(pred)/d(featpred)
   
   uniques = model["cats"][col]["uniques"].keys()
   
   #all the uniques . . .
   for u in uniques:
    finalGradient = sum((peoc*gradient)[inputDf[col].isin([u])])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    model["cats"][col][u] -= finalGradient*lr/len(inputDf)
   
   # . . . and "OTHER"
   finalGradient = sum((peoc*gradient)[~inputDf[col].isin(model["cats"][col]["uniques"].keys())])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
   model["cats"][col]["OTHER"] -= finalGradient*lr/len(inputDf)
 
 return model

def construct_mresp_model(inputDf, target, nrounds, lr, startingModels, grad=calculus.Poisson_grad):
 
 models = copy.deepcopy(startingModels)
 
 for i in range(nrounds):
  
  print("epoch: "+str(i)+"/"+str(nrounds))
  for model in models:
   apply_model.explain(model)
  
  preds=[]
  overallPred=pd.Series([0]*len(inputDf))
  for model in models:
   pred = apply_model.predict(inputDf, model)
   preds.append(pred) 
   overallPred += pred
  
  gradient = grad(overallPred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  for m in range(len(models)):
   model=models[m]
   pred=preds[m]
   for col in model["conts"]:
    effectOfCol = apply_model.get_effect_of_this_cont_col(inputDf, model, col)
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    totalAffectList=[0]*len(model["conts"][col])
    totalChangeList=[0]*len(model["conts"][col])
    
    #effect of outer range on starting point
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(inputDf[col]<model["conts"][col][0][0])] = 1 #d(featpred)/d(pt)
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    totalAffectList[0]+=sum(affectedness)
    totalChangeList[0]+=finalGradient*lr
    
    for i in range(len(model["conts"][col])-1):
     #define terms
     x = inputDf[col]
     x1 = model["conts"][col][i][0]
     x2 = model["conts"][col][i+1][0]
     #points are affected by what's ahead . . .
     affectedness = pd.Series([0]*len(inputDf))
     affectedness.loc[(x>=x1) & (x<x2)] = (x2 - x)/(x2 - x1) #d(featpred)/d(pt)
     finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     totalAffectList[i]+=sum(affectedness)
     totalChangeList[i]+=finalGradient
     #. . . and what's behind.
     affectedness = pd.Series([0]*len(inputDf))
     affectedness.loc[(x>=x1) & (x<x2)] = (x-x1)/(x2 - x1) #d(featpred)/d(pt)
     finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     totalAffectList[i+1]+=sum(affectedness)
     totalChangeList[i+1]+=finalGradient
    
    # effect of outer range on final point
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(inputDf[col]>=model["conts"][col][-1][0])] = 1 #d(featpred)/d(pt)
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    totalAffectList[i+1]+=sum(affectedness)
    totalChangeList[i+1]+=finalGradient
    
    for i in range(len(totalAffectList)):
     if totalAffectList[i]>0:
      models[m]["conts"][col][i][1]-= lr*totalChangeList[i]/totalAffectList[i]
   
   for col in model["cats"]:
    
    effectOfCol = apply_model.get_effect_of_this_cont_col(inputDf, model, col)
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    uniques = model["cats"][col]["uniques"].keys()
    
    #all the uniques . . .
    for u in uniques:
     finalGradient = sum((peoc*gradient)[inputDf[col].isin([u])])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     models[m]["cats"][col][u] -= finalGradient*lr/len(inputDf)
    
    # . . . and "OTHER"
    finalGradient = sum((peoc*gradient)[~inputDf[col].isin(model["cats"][col]["uniques"].keys())])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    models[m]["cats"][col]["OTHER"] -= finalGradient*lr/len(inputDf)
 
 return models


if __name__ == '__main__':
 df = pd.DataFrame({"x":[1,2,3],"y":[2,2,2]})
 model = {"BIG_C":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}
 newModel = construct_model(df, "y",50, 0.5, model)
 apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 model = {"BIG_C":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}
 newModel = construct_model(df, "y",50, 0.5, model)
 apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2],"y":[2,2]})
 model = {"BIG_C":1.0,"conts":{"x":[[1,1],[2,1]]}, "cats":[]}
 newModel = construct_model(df, "y",50, 0.5, model)
 apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1, 995,996,997,998,999,1000],"y":[2,2,2,2,2,2,2]})
 model = {"BIG_C":1.0,"conts":{"x":[[1,1],[1000,1]]}, "cats":[]}
 newModel = construct_model(df, "y",50, 0.5, model)
 apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3,4],"y":[1+1,2+1.2,3+1.4,4+1.6]})
 models = [{"BIG_C":2,"conts":{"x":[[1,1],[4,1]]}, "cats":[]},{"BIG_C":1,"conts":{"x":[[1,1],[4,1]]}, "cats":[]}]
 newModels = construct_mresp_model(df, "y",100, 0.5, models)
 for newModel in newModels:
  apply_model.explain(newModel)

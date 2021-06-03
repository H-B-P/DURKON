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
   
   #do the pt at the start
   affectedness = pd.Series([0]*len(inputDf))
   affectedness.loc[(inputDf[col]<model["conts"][col][0][0])] = 1 #d(featpred)/d(pt)
   totalAffectedness = sum(affectedness)
   if totalAffectedness>0:
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    model["conts"][col][0][1] -= finalGradient*lr/totalAffectedness
   
   for i in range(len(model["conts"][col])-1):
    #define terms
    x = inputDf[col]
    x1 = model["conts"][col][i][0]
    x2 = model["conts"][col][i+1][0]
    #affected by what's ahead . . .
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(x>=x1) & (x<x2)] = (x2 - x)/(x2 - x1) #d(featpred)/d(pt)
    totalAffectedness = sum(affectedness)
    if totalAffectedness>0:
     finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     model["conts"][col][i][1] -= finalGradient*lr/totalAffectedness
    #. . . and what's behind.
    affectedness = pd.Series([0]*len(inputDf))
    affectedness.loc[(x>=x1) & (x<x2)] = (x-x1)/(x2 - x1) #d(featpred)/d(pt)
    totalAffectedness = sum(affectedness)
    if totalAffectedness>0:
     finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     model["conts"][col][i+1][1] -= finalGradient*lr/totalAffectedness
  
   # effect of outer range on final point
   affectedness = pd.Series([0]*len(inputDf))
   affectedness.loc[(inputDf[col]>=model["conts"][col][-1][0])] = 1 #d(featpred)/d(pt)
   totalAffectedness = sum(affectedness)
   if totalAffectedness>0:
    finalGradient = sum(peoc*gradient*affectedness) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    model["conts"][col][-1][1] -= finalGradient*lr/totalAffectedness
  
  for col in model["cats"]:
   
   effectOfCol = apply_model.get_effect_of_this_cont_col(inputDf, model, col)
   
   peoc = pred/effectOfCol #d(pred)/d(featpred)
   
   uniques = model["cats"][col]["uniques"].keys()
   
   #all the uniques . . .
   for u in uniques:
    totalAffectedness = sum(inputDf[col].isin([u]))
    if totalAffecteness>0:
     finalGradient = sum((peoc*gradient)[inputDf[col].isin([u])])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
     model["cats"][col][u] -= finalGradient*lr/totalAffectedness
   
   # . . . and "OTHER"
   totalAffectedness = sum(~inputDf[col].isin(uniques))
   if totalAffecteness>0:
    finalGradient = sum((peoc*gradient)[~inputDf[col].isin(model["cats"][col]["uniques"].keys())])#d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    model["cats"][col]["OTHER"] -= finalGradient*lr/totalAffectedness
 
 return model

if __name__ == '__main__':
 #df = pd.DataFrame({"x":[1,2,3],"y":[2,2,2]})
 #model = {"BIG_C":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}
 #newModel = construct_model(df, "y",50, 0.5, model)
 #apply_model.explain(newModel)
 #
 #df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 #model = {"BIG_C":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}
 #newModel = construct_model(df, "y",200, 0.5, model)
 #apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2],"y":[2,2]})
 model = {"BIG_C":1.0,"conts":{"x":[[1,1],[2,1]]}, "cats":[]}
 newModel = construct_model(df, "y",200, 0.5, model)
 apply_model.explain(newModel)

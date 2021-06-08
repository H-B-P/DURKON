import pandas as pd
import numpy as np
import math
import copy

def gently_monotonize_this_cont_col(model, col, uponly=True):
 newModel = copy.deepcopy(model)
 for i in range(len(model["conts"][col])-1):
  if (uponly and (model["conts"][col][i+1][1]<model["conts"][col][i][1])) or ((not uponly) and (model["conts"][col][i+1][1]>model["conts"][col][i][1])):
   newModel["conts"][col][i][1] += (model["conts"][col][i+1][1]-model["conts"][col][i][1])/2.0
   newModel["conts"][col][i+1][1] += (model["conts"][col][i][1]-model["conts"][col][i+1][1])/2.0
 return newModel

def roughly_monotonize_this_cont_col(model, col, uponly=True):
 newModel = copy.deepcopy(model)
 for i in range(len(model["conts"][col])-1):
  if (uponly and (newModel["conts"][col][i+1][1]<newModel["conts"][col][i][1])) or ((not uponly) and (newModel["conts"][col][i+1][1]>newModel["conts"][col][i][1])):
   newModel["conts"][col][i+1][1] = newModel["conts"][col][i][1]
 return newModel

def monotonize_this_cont_col(model, col, uponly=True, iters=20):
 newModel = copy.deepcopy(model)
 for i in range(iters):
  newModel = gently_monotonize_this_cont_col(newModel, col, uponly)
 newModel = roughly_monotonize_this_cont_col(newModel, col, uponly)
 return newModel

if __name__ == '__main__':
 model = {"BIG_C":1.0,"conts":{"x":[[1,4],[2,3],[3,2],[4,1]]}, "cats":[]}
 print(model)
 print(gently_monotonize_this_cont_col(model,"x"))
 print(roughly_monotonize_this_cont_col(model,"x"))
 print(monotonize_this_cont_col(model,"x"))
 print("-")
 model = {"BIG_C":1.0,"conts":{"x":[[1,1],[2,2],[3,3],[4,2],[5,1],[6,2],[7,3]]}, "cats":[]}
 print(model)
 print(gently_monotonize_this_cont_col(model,"x"))
 print(roughly_monotonize_this_cont_col(model,"x"))
 print(monotonize_this_cont_col(model,"x"))

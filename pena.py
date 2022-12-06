import numpy as np
import pandas as pd

def move_to_default(x, l, defaultValue=1):
 if x>defaultValue:
  x=max(x-l, defaultValue)
 elif x<defaultValue:
  x=min(x+l, defaultValue)
 return x

def penalize_model(model, pen, defaultValue=1, specificPenas={}):
 
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

if __name__ == '__main__':
 model={"BASE_VALUE":4, "conts":{"cont1":[[1,1.4], [2,0.7]]}, "cats":{"cat1":{"OTHER":1.9, "uniques":{"a":1.05, "b":0.2}}}, "catcats":{"cat1 X cat2": {"OTHER":{"OTHER":1.1, "uniques":{"c":0.99, "d":0.21}}, "uniques":{"a":{"OTHER":1.1, "uniques":{"c":0.99, "d":0.21}}, "b":{"OTHER":1.1, "uniques":{"c":0.99, "d":0.21}}}}}}
 model = penalize_model(model, 0.1)
 print(model)
 
 
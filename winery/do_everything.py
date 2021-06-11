import numpy as np
import pandas as pd

import seg
import actual_modelling
import analytics
import util
import apply_model
import viz

PTS_PER_CONT=5

#==Load in==

df = pd.read_csv("data/winequality-red.csv", sep=";")

catCols  = []
contCols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar","chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]


#==Random Split==

trainDf = df.sample(frac = 0.5, random_state=1) 
testDf = df.drop(trainDf.index)

trainDf = trainDf.reset_index()
testDf = testDf.reset_index()

#==Prep==

print("PREPARING")

#Categoricals

def get_uniques_for_this_cat_col(inputDf, col, threshold=0):
 uniques = pd.unique(inputDf[col])
 passingUniques = []
 for unique in uniques:
  if (float(len(inputDf[inputDf[col]==unique]))/float(len(inputDf)))>=threshold:
   passingUniques.append(unique)
 return passingUniques

uniques={}

if True:
 for c in catCols:
  print(c)
  uniques[c] = get_uniques_for_this_cat_col(trainDf,c, 0.05)

#Segmentation

segPoints={}

for col in contCols:
 print(col)
 segPoints[col]=seg.default_seg(trainDf, col, PTS_PER_CONT, 0.01)

#==Actually model!===

models = [apply_model.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "quality",1,1)]#,apply_model.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "quality",1,0.4)]
#models = actual_modelling.construct_mresp_model(trainDf, "quality", 100, 0.01, models, {"contStraight":0.0005, "contGroup":0.0005})
#for i in range(len(models)):
# models[i] = apply_model.de_feat(models[i])
models = actual_modelling.construct_mresp_model(trainDf, "quality", 150, 0.01, models)

#==Visualize model==

for model in models:
 for col in model["conts"]:
  viz.draw_cont_pdp(model["conts"][col], 0.2, col)

#==Predict==

testDf["PREDICTED"] = 0
for model in models:
 testDf["PREDICTED"] += apply_model.predict(testDf, model)

print(testDf[["quality","PREDICTED"]])

#==Look at Predictions==

p, a = analytics.get_Xiles(testDf, "PREDICTED", "quality", 10)
print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])

#==Analyze (i.e. get summary stats)==

print("MAE")
print(util.round_to_sf(analytics.get_mae(testDf["PREDICTED"],testDf["quality"])))
print("RMSE")
print(util.round_to_sf(analytics.get_rmse(testDf["PREDICTED"],testDf["quality"])))
print("MEANS")
print(util.round_to_sf(testDf["PREDICTED"].mean()), util.round_to_sf(testDf["quality"].mean()))
print("DRIFT COEFF")
#print(analytics.get_drift_coeff(testDf["PREDICTED"],testDf["loss"]))
print(util.round_to_sf(analytics.get_drift_coeff_macro(p,a)))

import numpy as np
import pandas as pd

import seg
import actual_modelling
import analytics
import util
import apply_model

SEGS_PER_CONT=3

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

print(trainDf)
print(trainDf["fixed acidity"])

segPoints={}

for col in contCols:
 print(col)
 
 ratioList = seg.get_ratios(SEGS_PER_CONT)
 segPointList = []
 for ratio in ratioList:
  segpt = seg.get_segpt(trainDf, col, ratio)
  roundedSegpt = util.round_to_sf(segpt, 3)
  if roundedSegpt not in segPointList and (roundedSegpt>min(df[col])) and (roundedSegpt<max(df[col])):
   segPointList.append(roundedSegpt)
 segPointList.sort()
 segPoints[col]=[util.round_to_sf(min(trainDf[col]),3)]+segPointList+[util.round_to_sf(max(trainDf[col]),3)]

#==Actually model!===

model = apply_model.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "quality")
model = actual_modelling.construct_model(trainDf, "quality", 100, 0.01, model, {"contStraight":0.001, "contGroup":0.001})
model = apply_model.de_feat(model)
model = actual_modelling.construct_model(trainDf, "quality", 100, 0.01, model)

#==Predict==

testDf["PREDICTED"]=apply_model.predict(testDf, model)

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

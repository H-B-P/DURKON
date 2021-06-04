import pandas as pd
import numpy as np
import math

def get_rmse(pred, act):
 err = pred-act
 return math.sqrt(sum(err*err)/len(err))

def get_mae(pred,act):
 err = pred-act
 return sum(abs(err))/len(err)

def get_means(pred,act):
 return sum(pred)/len(pred), sum(act)/len(act)

def get_drift_coeff_macro(predXiles, actXiles):
 avePred = sum(predXiles)/len(predXiles)
 aveAct = sum(actXiles)/len(actXiles)
 
 numerator=0
 denominator=0
 
 for i in range(len(predXiles)):
  numerator+=(predXiles[i]-avePred)*(actXiles[i]-aveAct)
  denominator+=(predXiles[i]-avePred)*(predXiles[i]-avePred)
 
 return numerator/denominator

def get_Xiles(df, predCol, actCol, X=10):
 cdf = df.copy()
 cdf = cdf.sort_values([predCol, actCol])
 cdf = cdf.reset_index(drop=True)
 preds = []
 acts = []
 for i in range(X):
  lowerlim = int(((i)*len(cdf))/X)
  upperlim = int(((i+1)*len(cdf))/X)
  subset = cdf[lowerlim:upperlim]
  preds.append(sum(subset[predCol])/float(len(subset)))
  acts.append(sum(subset[actCol])/float(len(subset)))
 return preds, acts

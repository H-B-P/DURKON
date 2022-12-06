import pandas as pd
import numpy as np
import math

def get_weighted_gini(df, predCol, actCol, weightCol):
 df['meanAct'] = sum(df[actCol]*df[weightCol])/sum(df[weightCol])
 df['invPred'] = -df[predCol]
 df = df.sort_values(['invPred', actCol])
 df['diff'] = df[actCol]-df['meanAct']
 df['gap'] = (df['diff']*df[weightCol]).cumsum()
 df['base'] = (df['meanAct']*df[weightCol]).cumsum()
 return sum(df['gap']*df[weightCol])/sum(df['base']*df[weightCol])

def get_weighted_Xiles(df, predCol, actCol, weightCol, X=10):
 df['invAct'] = -df[actCol]
 df = df.sort_values([predCol, 'invAct'])
 totWeight = sum(df[weightCol])
 df['cumWeight'] = df[weightCol].cumsum()
 p=[]
 a=[]
 for i in range(X):
  subDf = df[df['cumWeight']>(i*totWeight/X)][df['cumWeight']<=((i+1)*totWeight/X)]
  p.append(sum(subDf[predCol]*subDf[weightCol])/sum(subDf[weightCol]))
  a.append(sum(subDf[actCol]*subDf[weightCol])/sum(subDf[weightCol]))
 return p, a

def get_gini(df, predCol, actCol):
 df["w8"] = 1
 return get_weighted_gini(df, predCol, actCol, 'w8')

def get_Xiles(df, predCol, actCol, X=10):
 df["w8"] = 1
 return get_weighted_Xiles(df, predCol, actCol, 'w8', X)

#---

def get_weighted_MAE(df, predCol, actCol, weightCol):
 return ((df[predCol]-df[actCol]).abs()*df[weightCol]).sum()/sum(weightCol)

def get_weighted_RMSE(df, predCol, actCol, weightCol):
 return math.sqrt(((df[predCol]-df[actCol])*(df[predCol]-df[actCol])*df[weightCol]).sum()/sum(weightCol))

def get_weighted_MPE(df, predCol, actCol, weightCol): #Yes, I know the standard definition uses Actual as the denominator. The standard definition is wrong.
 PE = 100*(df[predCol]-df[actCol])/df[predCol]
 return (PE.abs()*df[weightCol]).sum()/sum(weightCol)

def get_weighted_RMSPE(df, predCol, actCol, weightCol): 
 PE = 100*(df[predCol]-df[actCol])/df[predCol]
 return math.sqrt((PE*PE*df[weightCol]).sum()/sum(weightCol))

def get_MAE(df, predCol, actCol):
 df["w8"] = 1
 return get_weighted_MAE(df, predCol, actCol, "w8")

def get_RMSE(df, predCol, actCol):
 df["w8"] = 1
 return get_weighted_RMSE(df, predCol, actCol, "w8")

def get_MPE(df, predCol, actCol):
 df["w8"] = 1
 return get_weighted_MPE(df, predCol, actCol, "w8")

def get_RMSPE(df, predCol, actCol):
 df["w8"] = 1
 return get_weighted_RMSPE(df, predCol, actCol, "w8")

#---

def get_custom_metric(df, predCol, actCol, OF=[[-1000,0],[-50,0.1],[-20,0.2],[0,1],[10,0],[1000,-99]]):
 overestPercent = 100*(df[predCol]-df[actCol])/df[actCol]
 

if __name__ == '__main__':
 df = pd.DataFrame({"P":[1,2,3,4,5,7,6,8,9],"A":[1,2,3,4,5,6,8,7,9],"W":[1,2,3,4,5,6,7,8,9]})
 print(get_weighted_gini(df, 'P','A','W'))
 print(get_weighted_gini(df, 'A','A','W'))
 print(get_weighted_gini(df, 'P','A','W')/get_weighted_gini(df, 'A','A','W'))
 
 print(get_weighted_Xiles(df, 'P','A','W', 3))

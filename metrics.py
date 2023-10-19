import pandas as pd
import numpy as np
import math

def get_weighted_gini(df, predCol, actCol, weightCol, tiebreaks="pessimistic"):
 df['meanAct'] = sum(df[actCol]*df[weightCol])/sum(df[weightCol])
 
 if tiebreaks=="pessimistic":
  df = df.sort_values([predCol, actCol], ascending=[False,True])
 elif tiebreaks=="optimistic":
  df = df.sort_values([predCol, actCol], ascending=[False,False])
 else:
  df = df.sort_values(predCol, ascending=False)
 
 df['diff'] = df[actCol]-df['meanAct']
 df['gap'] = (df['diff']*df[weightCol]).cumsum()
 df['base'] = (df['meanAct']*df[weightCol]).cumsum()
 return sum(df['gap']*df[weightCol])/sum(df['base']*df[weightCol])

def get_weighted_Xiles(df, predCol, actCol, weightCol, X=10, tiebreaks="pessimistic"):
 if tiebreaks=="pessimistic":
  df = df.sort_values([predCol, actCol], ascending=[True,False])
 elif tiebreaks=="optimistic":
  df = df.sort_values([predCol, actCol], ascending=[True,True])
 else:
  df = df.sort_values(predCol, ascending=True)
 
 totWeight = sum(df[weightCol])
 df['cumWeight'] = df[weightCol].cumsum()
 p=[]
 a=[]
 for i in range(X):
  subDf = df[df['cumWeight']>(i*totWeight/X)].reset_index(drop=True)
  subDf = subDf[subDf['cumWeight']<=((i+1)*totWeight/X)]
  p.append(sum(subDf[predCol]*subDf[weightCol])/sum(subDf[weightCol]))
  a.append(sum(subDf[actCol]*subDf[weightCol])/sum(subDf[weightCol]))
 return p, a

def get_gini(df, predCol, actCol, tiebreaks="pessimistic"):
 df["w8"] = 1
 return get_weighted_gini(df, predCol, actCol, 'w8', tiebreaks)

def get_Xiles(df, predCol, actCol, X=10, tiebreaks="pessimistic"):
 df["w8"] = 1
 return get_weighted_Xiles(df, predCol, actCol, 'w8', X, tiebreaks)

#---

def get_weighted_MAE(df, predCol, actCol, weightCol):
 return ((df[predCol]-df[actCol]).abs()*df[weightCol]).sum()/sum(df[weightCol])

def get_weighted_RMSE(df, predCol, actCol, weightCol):
 return math.sqrt(((df[predCol]-df[actCol])*(df[predCol]-df[actCol])*df[weightCol]).sum()/sum(df[weightCol]))

def get_weighted_MPE(df, predCol, actCol, weightCol): #Yes, I know the standard definition uses Actual as the denominator. The standard definition is wrong.
 PE = 100*(df[predCol]-df[actCol])/df[predCol]
 return (PE.abs()*df[weightCol]).sum()/sum(df[weightCol])

def get_weighted_RMSPE(df, predCol, actCol, weightCol): 
 PE = 100*(df[predCol]-df[actCol])/df[predCol]
 return math.sqrt((PE*PE*df[weightCol]).sum()/sum(df[weightCol]))

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
 

if __name__ == '__main__':
 df = pd.DataFrame({"P":[1,2,3,4,5,7,6,8,9],"A":[1,2,3,4,5,6,8,7,9],"W":[1,2,3,4,5,6,7,8,9]})
 print(get_weighted_gini(df, 'P','A','W'))
 print(get_weighted_gini(df, 'A','A','W'))
 print(get_weighted_gini(df, 'P','A','W')/get_weighted_gini(df, 'A','A','W'))
 
 print(get_weighted_Xiles(df, 'P','A','W', 3))
 
 
 df = pd.DataFrame({"P":[1,3,3,3,5],"A":[1,2,4,3,5]})
 print(get_gini(df,'P','A', "pessimistic"))
 print(get_gini(df,'P','A', "optimistic"))
 print(get_gini(df,'P','A', "whatever"))
 
 print(get_Xiles(df,'P','A', 5, "pessimistic"))
 print(get_Xiles(df,'P','A', 5, "optimistic"))
 print(get_Xiles(df,'P','A', 5, "whatever"))
 

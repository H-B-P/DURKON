import pandas as pd
import numpy as np

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

if __name__ == '__main__':
 df = pd.DataFrame({"P":[1,2,3,4,5,7,6,8,9],"A":[1,2,3,4,5,6,8,7,9],"W":[1,2,3,4,5,6,7,8,9]})
 print(get_weighted_gini(df, 'P','A','W'))
 print(get_weighted_gini(df, 'A','A','W'))
 print(get_weighted_gini(df, 'P','A','W')/get_weighted_gini(df, 'A','A','W'))
 
 print(get_weighted_Xiles(df, 'P','A','W', 3))

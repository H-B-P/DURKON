import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

def get_cont_pdp_prevalences(df, col, intervals=10, weightCol=None):
 cdf= df.copy()
 
 if type(intervals)==type([1,2,3]):
  intervs=intervals
 else:
  gap=(max(df[col])-min(df[col]))/float(intervals)
  print(min(df[col]), max(df[col]), gap)
  intervs=list(np.arange(min(df[col]), max(df[col])+gap, gap))
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 prevs=[]
 
 for i in range(len(intervs)-1):
  loInt = intervs[i]
  hiInt = intervs[i+1]
  if i==(len(intervs)-2):
   prevs.append(sum(cdf[(cdf[col]<=hiInt) & (cdf[col]>=loInt)]["weight"]))
  else:
   prevs.append(sum(cdf[(cdf[col]<hiInt) & (cdf[col]>=loInt)]["weight"]))
 
 return intervs, prevs

def get_cat_pdp_prevalences(df, col, threshold=0.05, weightCol=None):
 cdf= df.copy()
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 uniques = pd.unique(cdf[col])
 opDict = {}
 totalWeight = float(sum(cdf["weight"]))
 
 for unique in uniques:
  specWeight = float(sum(cdf[cdf[col]==unique]["weight"]))
  if (specWeight/totalWeight)>=threshold:
   opDict[unique] = specWeight
 
 opDict["OTHER"] = sum(cdf[~cdf[col].isin(opDict)]["weight"])
 
 return opDict



def draw_cont_pdp(pts, targetSpan=0, name="graph"):
 X = [pt[0] for pt in pts]
 Y = [pt[1] for pt in pts]
 layout = {
  "yaxis": {
    "range": [min(min(Y), 1-targetSpan), max(max(Y), 1+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Scatter(x=X, y=Y), layout=layout)
 
 plotly.offline.plot(fig, filename='graphs/'+name+'.html')

def draw_cat_pdp(dyct, targetSpan=0, name="graph"):
 
 X=[]
 Y=[]
 for thing in dyct["uniques"]:
  X.append(thing)
  Y.append(dyct["uniques"][thing])
 X.append("OTHER")
 Y.append(dyct["OTHER"])
 
 print(X,Y)
 
 layout = {
  "yaxis": {
    "range": [0, max(max(Y), 1+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Bar(x=X, y=Y), layout=layout)
 
 plotly.offline.plot(fig, filename='graphs/'+name+'.html')

if __name__=="__main__":
 #exampleCont = [[1,1.4],[2,1.6],[3,0.4]]
 #draw_cont_pdp(exampleCont)
 exampleCat = {"uniques":{"wstfgl":1.05, "florpalorp":0.92, "turlingdrome":0.99}, "OTHER":1.04}
 draw_cat_pdp(exampleCat)

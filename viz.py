import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

from . import misc

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



def draw_cont_pdp(pts, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", folder="graphs"):
 X = [pt[0] for pt in pts]
 Y = [pt[1] for pt in pts]
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Scatter(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image(folder+"/"+name+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  fig.write_image(folder+"/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)
 




def draw_cat_pdp(dyct, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", folder="graphs", otherName="OTHER"):
 
 X=[]
 Y=[]
 for thing in dyct["uniques"]:
  X.append(thing)
  Y.append(dyct["uniques"][thing])
 if (dyct["OTHER"]!=defaultValue):
  X.append(otherName)
  Y.append(dyct["OTHER"])
 
 #print(X,Y)
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Bar(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image(folder+"/"+name+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  fig.write_image(folder+"/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)

def draw_catcat_pdp(catcat, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", cat1="", cat2="", shapes=['/', '\\', 'x', '-', '|', '+', '.'], colors = ['red','orange','yellow','green','blue','indigo','violet'], folder="graphs", otherName="OTHER"):
 
 bars=[]
 
 styleno=0
 
 for thing1 in catcat["uniques"]:
  X2=[]
  Y=[]
  for thing2 in catcat["uniques"][thing1]['uniques']:
   X2.append(thing2)
   Y.append(catcat["uniques"][thing1]['uniques'][thing2])
  if (catcat["uniques"][thing1]["OTHER"]!=defaultValue):
   X2.append(otherName)
   Y.append(catcat["uniques"][thing1]["OTHER"])
  bars.append(go.Bar(name=thing1, x=X2, y=Y, marker_color=colors[styleno], marker_pattern={'shape':shapes[styleno]}))
  styleno+=1
 
 X2=[]
 Y=[]
 botherWithOther=False
 for thing2 in catcat["OTHER"]["uniques"]:
  X2.append(thing2)
  Y.append(catcat["OTHER"]["uniques"][thing2])
  if (catcat["OTHER"]["uniques"][thing2]!=defaultValue):
   botherWithOther=True
 X2.append(otherName)
 Y.append(catcat["OTHER"]["OTHER"])
 if catcat["OTHER"]["OTHER"]!=defaultValue:
  botherWithOther=True
 if botherWithOther:
  bars.append(go.Bar(name=otherName, x=X2, y=Y, marker_color='gray', marker_pattern={'shape':''}))
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=bars, layout=layout)
 
 fig.update_layout(legend_title_text = cat1)
 fig.update_xaxes(title_text = cat2)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name, xaxis_title=cat2, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model, xaxis_title=cat2, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image(folder+"/"+name+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  fig.write_image(folder+"/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)

def draw_catcont_pdp(catcont, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", cat="", cont="", colors = ['red','orange','yellow','green','blue','indigo','violet'], folder="graphs", otherName="OTHER"):
 
 lines=[]
 
 styleno=0
 
 botherWithOther=False
 for pt in catcont["OTHER"]:
  if (pt[1]!=defaultValue):
   botherWithOther=True
 
 Y = []
 for thing in catcont['uniques']:
  Y=Y+[pt[1] for pt in catcont['uniques'][thing]]
 if botherWithOther:
  Y=Y+[pt[1] for pt in catcont['OTHER']]
 
 for thing in catcont['uniques']:
  lines.append(go.Scatter(name=thing, x=[pt[0] for pt in catcont['uniques'][thing]], y=[pt[1] for pt in catcont['uniques'][thing]], marker_symbol=styleno+1, marker_color=colors[styleno], marker_size=10))
  styleno+=1
 if botherWithOther:
  lines.append(go.Scatter(name=otherName, x=[pt[0] for pt in catcont['OTHER']], y=[pt[1] for pt in catcont['OTHER']], marker_symbol=0, marker_color="gray", marker_size=10))
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, 0, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=lines, layout=layout)
 
 fig.update_layout(legend_title_text = cat)
 fig.update_xaxes(title_text = cont)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image(folder+"/"+name+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  fig.write_image(folder+"/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)
 

 
def draw_contcont_pdp(contcont, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", cont1="", cont2="", lws=[5,7,9,11,13,15,17], colors = ['#00F','#44F','#77F','#99F','#AAF','#CCF'], folder="graphs"):
 lines = []
 
 styleno=0
 
 Y=[]
 for pt in contcont:
  Y=Y+[ptpt[1] for ptpt in pt[1]]
 
 for pt in contcont:
  lines.append(go.Scatter(name=pt[0], x=[ptpt[0] for ptpt in pt[1]], y=[ptpt[1] for ptpt in pt[1]], line_width = 1+lws[styleno]/4, marker_size = lws[styleno], marker_color=colors[styleno]))#'blue'))
  styleno+=1
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=lines, layout=layout)
 
 fig.update_layout(legend_title_text = cont1)
 fig.update_xaxes(title_text = cont2)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name, xaxis_title=cont2, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model, xaxis_title=cont2, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image(folder+"/"+name+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  fig.write_image(folder+"/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)
 
def draw_contcont_pdp_3D(contcont, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", cont1="", cont2="", folder="graphs"):
 X1=[]
 X2=[]
 Y=[]
 
 for pt in contcont:
  X1=X1+[pt[0] for ptpt in pt[1]]
  X2=X2+[ptpt[0] for ptpt in pt[1]]
  Y=Y+[ptpt[1] for ptpt in pt[1]]
 
 layout = {
  "zaxis": {
    "range": [min(min(Y)-leeway, defaultValue-targetSpan), max(max(Y)+leeway, defaultValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=[go.Mesh3d(x=X1, y=X2, z=Y, opacity=0.5)])
 
 fig.update_layout(scene = dict(
                    xaxis_title=cont1,
                    yaxis_title=cont2,
                    zaxis_title=ytitle))
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model)
 
 if (model==0):
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)


def get_cont_inputs(cont, detail=1):
 op=[]
 for i in range(len(cont)-1):
  for j in range(detail):
   op.append(float(cont[i][0]*(detail-j) + cont[i+1][0]*j)/detail)
 op.append(cont[-1][0])
 return op

def draw_contcont_pdp_heatmap(contcont, targetSpan=0, name="graph", model=0, defaultValue=1, leeway=0.05, ytitle="Relativity", cont1="", cont2="", folder="graphs", detail=10):
 X1=[]
 X2=[]
 Y=[]
 
 X1 = get_cont_inputs(contcont, detail)
 X2 = get_cont_inputs(contcont[0][1], detail)
 
 #for pt in contcont:
 # X1.append(pt[0])
 #for pt in contcont[0][1]:
 # X2.append(pt[0])
 
 for i in range(len(X1)-1):
  newYs=[]
  for j in range(len(X2)-1):
   newYs.append(misc.get_effect_of_this_contcont_on_single_input((X1[i]+X1[i+1])/2,(X2[j]+X2[j+1])/2, contcont))
  Y.append(newYs)
 
 Ymin = min([min(l) for l in Y])
 Ymax = max([max(l) for l in Y])
 
 fig = go.Figure(data=[go.Heatmap(x=X1, y=X2, z=Y, type='heatmap', colorscale="bluered", zmin =min(Ymin-leeway, defaultValue-targetSpan), zmax = max(Ymax+leeway, defaultValue+targetSpan), zmid=defaultValue, colorbar=dict(title=ytitle))])
 
 
 fig.update_layout( xaxis=dict(title=cont1),
                    yaxis=dict(title=cont2))
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP for "+name)
  else:
   fig.update_layout(title="PDP for "+name+", model "+model)
 
 if (model==0):
  plotly.offline.plot(fig, filename=folder+"/"+name+'.html', auto_open = False)
 else:
  plotly.offline.plot(fig, filename=folder+"/"+name+'__'+model+'.html', auto_open = False)
 

if __name__=="__main__":
 
 exampleCat = {"uniques":{"Red":1.05, "Blue":0.92, "Green":0.99, "Black":1.12}, "OTHER":1.04}
 draw_cat_pdp(exampleCat, 0.5, "Vehicle_Colour", ytitle="Multiplier")
 
 exampleCont = [[0,1.19],[1,1.01],[10000,0.93],[50000,0.81]]
 draw_cont_pdp(exampleCont, 0.5, "Vehicle_Odometer_Reading", ytitle="Multiplier")
 
 exampleCatCat = {'uniques':{'Britain':{'uniques':{'Four-Wheel Drive':1.1,'Two-Wheel Drive':1.2},'OTHER':1.3},'France':{'uniques':{'Four-Wheel Drive':1.4,'Two-Wheel Drive':1.0},'OTHER':1.6}},'OTHER':{'uniques':{'Four-Wheel Drive':1.7,'Two-Wheel Drive':1.35},'OTHER':1.9}}
 draw_catcat_pdp(exampleCatCat, 0.5, "Vehicle_Nation_of_Origin X Vehicle_Drive_Type", cat1="Vehicle_Nation_of_Origin", cat2="Vehicle_Drive_Type", ytitle="Multiplier")
 
 exampleCatCont = {"uniques":{"Ford":[[1000,1.1],[2000,1.2],[3500,1.3]],"Hyundai":[[1000,1.4],[2000,1.5],[3500,1.6]]},"OTHER":[[1000,1.7],[2000,1.8],[3500,1.9]]}
 draw_catcont_pdp(exampleCatCont, 0.5, "Vehicle_Make X Vehicle_Mass", cat="Vehicle_Make", cont="Vehicle_Mass", ytitle="Multiplier")
 
 exampleContCont = [[0,[[18,1.1],[29,1.2],[75,1.3]]],[10,[[18,1.0],[29,1.1],[75,1.1]]],[20,[[18,0.9],[29,1.4],[75,1.2]]]]
 draw_contcont_pdp(exampleContCont, 0.5, "Vehicle_Age X Owner_Age", cont1="Vehicle_Age", cont2="Owner_Age", ytitle="Multiplier")
 draw_contcont_pdp_3D(exampleContCont, 0.5, "Vehicle_Age X Owner_Age (alternative 3D version)", cont1="Vehicle_Age", cont2="Owner_Age", ytitle="Multiplier")
 draw_contcont_pdp_heatmap(exampleContCont, 0.5, "Vehicle_Age X Owner_Age (alternative heatmap version)", cont1="Vehicle_Age", cont2="Owner_Age", ytitle="Multiplier")
 
 

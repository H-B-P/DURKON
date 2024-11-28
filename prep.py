import pandas as pd
import numpy as np

import copy

def prep_model(df, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, contEdge=0.01, defaultValue=1, weightCol=None, bandConts=False):
 
 if weightCol==None:
  model={"BASE_VALUE":df[resp].mean(), "featcomb":"mult"}
 else:
  model={"BASE_VALUE":sum(df[resp]*df[weightCol])/sum(df[weightCol]), "featcomb":"mult"}
 
 for cat in cats:
  model = add_cat_to_model(model, df, cat, catMinPrev, defaultValue, weightCol)
 for cont in conts:
  if bandConts:
   model = add_banded_cont_to_model(model, df, cont, contTargetPts, contEdge, defaultValue, weightCol)
  else:
   model = add_cont_to_model(model, df, cont, contTargetPts, contEdge, defaultValue, weightCol)
 
 return model


def get_cat_feat(df, cat, catMinPrev=0.01, defaultValue=1, weightCol=None):
 
 if weightCol==None:
  df["WEIGHT_COL"]=1
 else:
  df["WEIGHT_COL"] = df[weightCol]
 
 sw = sum(df["WEIGHT_COL"])
 
 feat = {"OTHER":defaultValue, "uniques":{}}
 
 for u in df[cat].unique():
  if sum(df[df[cat]==u]["WEIGHT_COL"]/sw)>catMinPrev:
   feat['uniques'][u]=defaultValue
 
 return feat

def add_cat_to_model(model, df, cat, catMinPrev=0.01, defaultValue=1, weightCol=None):
 
 if "cats" not in model:
  model["cats"]={}
 model['cats'][cat] = get_cat_feat(df,cat,catMinPrev, defaultValue, weightCol)
 
 return model



def get_cont_feat(df, cont, contTargetPts=5, contEdge=0.01, defaultValue=1, weightCol=None):
 
 if weightCol==None:
  df["WEIGHT_COL"]=1
 else:
  df["WEIGHT_COL"] = df[weightCol]
 
 df = df[~df[cont].isna()]
 
 sw = sum(df["WEIGHT_COL"])
 
 inpts = []
 for i in range(contTargetPts):
  inpts.append(contEdge+(1-contEdge*2)*i/(contTargetPts-1))
 
 df = df.sort_values(by=cont).reset_index()
 df["CSWC"]=df["WEIGHT_COL"].cumsum()
 
 pts=[]
 feat=[]
 
 for inpt in inpts:
  newpt = min(df[df["CSWC"]>=(inpt*sw)][cont])
  if newpt not in pts:
   pts.append(newpt)
 
 for pt in pts:
  feat.append([pt,defaultValue])
 
 return feat

def add_cont_to_model(model, df, cont, contTargetPts=5, contEdge=0.01, defaultValue=1, weightCol=None):

 if "conts" not in model:
  model["conts"]={}
 model['conts'][cont] = get_cont_feat(df,cont,contTargetPts, contEdge, defaultValue, weightCol)
 
 return model



def get_contcont_feat(df, cont1, cont2, contTargetPts1=3, contTargetPts2=3, contEdge1=0.05, contEdge2=0.05, defaultValue=1, weightCol=None):
 
 feat1 = get_cont_feat(df, cont1, contTargetPts1, contEdge1, defaultValue, weightCol)
 feat2 = get_cont_feat(df, cont2, contTargetPts2, contEdge2, defaultValue, weightCol)
 
 for i in range(len(feat1)):
  feat1[i][1]=copy.deepcopy(feat2)
 
 return feat1

def add_contcont_to_model(model, df, cont1, cont2, contTargetPts1=3, contTargetPts2=3, contEdge1=0.05, contEdge2=0.05, defaultValue=1, weightCol=None, replace=False):
 
 if "contconts" not in model:
  model["contconts"]={}
 model['contconts'][cont1 + " X " + cont2] = get_contcont_feat(df, cont1, cont2, contTargetPts1, contTargetPts2, contEdge1, contEdge2, defaultValue, weightCol)
 
 if replace:
  if "conts" in model:
   if cont1 in model["conts"]:
    del model["conts"][cont1]
   if cont2 in model["conts"]:
    del model["conts"][cont2]
 
 return model



def get_catcont_feat(df, cat, cont, catMinPrev=0.1, contTargetPts=3, contEdge=0.05, defaultValue=1, weightCol=None):
 
 feat1 = get_cat_feat(df, cat, catMinPrev, defaultValue, weightCol)
 feat2 = get_cont_feat(df, cont, contTargetPts, contEdge, defaultValue, weightCol)
 
 feat1["OTHER"]=copy.deepcopy(feat2)
 
 for u in feat1["uniques"]:
  feat1["uniques"][u]=copy.deepcopy(feat2)
 
 return feat1

def add_catcont_to_model(model, df, cat, cont, catMinPrev=0.1, contTargetPts=3, contEdge=0.05, defaultValue=1, weightCol=None, replace=False):
 
 if "catconts" not in model:
  model["catconts"]={}
 model['catconts'][cat + " X " + cont] = get_catcont_feat(df, cat, cont, catMinPrev, contTargetPts, contEdge, defaultValue, weightCol)
 
 if replace:
  if "cats" in model:
   if cat in model["cats"]:
    del model["cats"][cat]
  if "conts" in model:
   if cont in model["conts"]:
    del model["conts"][cont]
 
 return model



def get_catcat_feat(df, cat1, cat2, catMinPrev1=0.1, catMinPrev2=0.1, defaultValue=1, weightCol=None):
 
 feat1 = get_cat_feat(df, cat1, catMinPrev1, defaultValue, weightCol)
 feat2 = get_cat_feat(df, cat2, catMinPrev2, defaultValue, weightCol)
 
 feat1["OTHER"]=copy.deepcopy(feat2)
 
 for u in feat1["uniques"]:
  feat1["uniques"][u]=copy.deepcopy(feat2)
 
 return feat1

def merget_catcat_feat(model, cat1, cat2, defaultValue=1):
 
 feat1 = copy.deepcopy(model["cats"][cat1])
 feat2 = copy.deepcopy(model["cats"][cat2])
 
 feat1["OTHER"]=copy.deepcopy(feat2)
 
 for u in feat1["uniques"]:
  feat1["uniques"][u]=copy.deepcopy(feat2)
 
 return feat1

def add_catcat_to_model(model, df, cat1, cat2, catMinPrev1=0.1, catMinPrev2=0.1, defaultValue=1, weightCol=None, replace=False, merget=False):
 
 if "catcats" not in model:
  model["catcats"]={}
 if merget:
  model['catcats'][cat1 + " X " + cat2] = merget_catcat_feat(model, cat1, cat2, defaultValue)
 else:
  model['catcats'][cat1 + " X " + cat2] = get_catcat_feat(df, cat1, cat2, catMinPrev1, catMinPrev2, defaultValue, weightCol)
 
 if replace:
  if "cats" in model:
   if cat1 in model["cats"]:
    del model["cats"][cat1]
   if cat2 in model["cats"]:
    del model["cats"][cat2]
 
 return model

#Bandings . . .

def get_banded_cont_feat(df, cont, contTargetPts=9, contEdge=0.1, defaultValue=1, weightCol=None):
 if weightCol==None:
  df["WEIGHT_COL"]=1
 else:
  df["WEIGHT_COL"] = df[weightCol]
 
 df = df[~df[cont].isna()]
 
 sw = sum(df["WEIGHT_COL"])
 
 inpts = []
 for i in range(contTargetPts):
  inpts.append(contEdge+(1-contEdge*2)*i/(contTargetPts-1))
 
 df = df.sort_values(by=cont).reset_index()
 df["CSWC"]=df["WEIGHT_COL"].cumsum()
 
 pts=[]
 feat={"uniques":{},"OTHER":defaultValue}
 
 for inpt in inpts:
  newpt = min(df[df["CSWC"]>=(inpt*sw)][cont])
  if newpt not in pts:
   pts.append(newpt)
 
 feat["uniques"]["< "+str(pts[0])]=defaultValue
 for i in range(len(pts)-1):
  feat["uniques"][str(pts[i])+" to "+str(pts[i+1])]=defaultValue
 feat["uniques"][">= "+str(pts[-1])]=defaultValue
 
 return feat, pts

def add_banded_cont_to_model(model, df, cont, contTargetPts=9, contEdge=0.1, defaultValue=1, weightCol=None):
 
 if "cats" not in model:
  model["cats"] = {}
 if "bandings" not in model:
  model["bandings"] = {}
 c,b = get_banded_cont_feat(df,cont, contTargetPts, contEdge, defaultValue, weightCol)
 model["cats"][cont]=c
 model["bandings"][cont]=b
 return model

def band_df(df, model):
 bandedDf = df.copy()
 
 if "bandings" in model:
  for col in model["bandings"]:
   pts = model["bandings"][col]
   bandedDf.loc[df[col]<pts[0],col] = "< "+str(pts[0])
   for i in range(len(model["bandings"][col])-1):
    bandedDf.loc[(df[col]>=pts[i])&(df[col]<pts[i+1]),col] = str(pts[i])+" to "+str(pts[i+1])
   bandedDf.loc[df[col]>=pts[-1],col] = ">= "+str(pts[-1])
 
 return bandedDf

if __name__ == '__main__':
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 print(get_cont_feat(df, "cont1", 5, 0.001, 1))
 
 
